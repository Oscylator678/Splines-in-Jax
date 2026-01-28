import jax.numpy as jnp
from jax import jit, vmap, grad
from functools import partial
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg

@staticmethod
def _bspline_basis(x, k, knots, degree):
    # Standard recursive Cox-de Boor
    if degree == 0:
        active = (x >= knots[k]) & (x < knots[k + 1])
        return jnp.where(active, 1.0, 0.0)

    denom1 = knots[k + degree] - knots[k]
    safe_denom1 = jnp.where(denom1 == 0, 1.0, denom1)
    term1_val = ((x - knots[k]) / safe_denom1) * _bspline_basis(x, k, knots, degree - 1)
    term1 = jnp.where(denom1 > 0, term1_val, 0)

    denom2 = knots[k + degree + 1] - knots[k + 1]
    safe_denom2 = jnp.where(denom2 == 0, 1.0, denom2)
    term2_val = ((knots[k + degree + 1] - x) / safe_denom2) * _bspline_basis(x, k + 1, knots, degree - 1)
    term2 = jnp.where(denom2 > 0, term2_val, 0.0)

    return term1 + term2



#Penalized B-Splines
@partial(jit, static_argnums=(5, 6))
def _fit_solver_kernel_penalized(x, y, z, x_knots, y_knots, degree, maxiter, regularization):
    x = jnp.ravel(x)
    y = jnp.ravel(y)
    z = jnp.ravel(z)

    N = x.shape[0]
    n_bx = x_knots.shape[0] - degree - 1
    n_by = y_knots.shape[0] - degree - 1
    M = n_bx * n_by

    # 1. Local Support Lookup
    idx_x = jnp.clip(jnp.searchsorted(x_knots, x, side='right') - 1, degree, n_bx - 1)
    idx_y = jnp.clip(jnp.searchsorted(y_knots, y, side='right') - 1, degree, n_by - 1)

    # (Helper function inside to ensure scope visibility)
    def get_active_basis(val, knot_idx, knots):
        active_indices = knot_idx - jnp.arange(degree, -1, -1)
        # Assuming _bspline_basis is available in scope or imported
        results = vmap(lambda k: _bspline_basis(val, k, knots, degree))(active_indices)

        is_end = jnp.isclose(val, knots[-1], atol=1e-10)
        end_mask = jnp.arange(degree + 1) == degree
        results = jnp.where(is_end, jnp.where(end_mask, 1.0, 0.0), results)
        return results, active_indices

    bx_vals, bx_inds = vmap(lambda v, i: get_active_basis(v, i, x_knots))(x, idx_x)
    by_vals, by_inds = vmap(lambda v, i: get_active_basis(v, i, y_knots))(y, idx_y)

    # 2. Build Sparse Matrix
    def compute_triplets(b_x, b_y, idx_x, idx_y):
        w_block = jnp.outer(b_y, b_x).flatten()
        iy_grid, ix_grid = jnp.meshgrid(idx_y, idx_x, indexing='ij')
        global_indices = iy_grid * n_bx + ix_grid
        return global_indices.flatten(), w_block

    all_indices, all_weights = vmap(compute_triplets)(bx_vals, by_vals, bx_inds, by_inds)

    row_indices = jnp.repeat(jnp.arange(N), (degree + 1) ** 2)
    A_sparse = sparse.BCOO(
        (all_weights.flatten(), jnp.column_stack((row_indices, all_indices.flatten()))),
        shape=(N, M)
    )

    # --- THE FIX: P-Spline Regularization ---
    # Define a function that computes the "roughness" of the surface
    def roughness_penalty(c_flat):
        # Reshape to grid (n_by rows, n_bx cols)
        c_grid = c_flat.reshape(n_by, n_bx)

        # Discrete 2nd derivative (D^T D equivalent)
        # This penalizes (c[i+1] - 2c[i] + c[i-1])^2
        diff_y = jnp.diff(c_grid, n=2, axis=0)  # Smoothness along Y
        diff_x = jnp.diff(c_grid, n=2, axis=1)  # Smoothness along X

        # P-Spline Objective: lambda * (||D_y c||^2 + ||D_x c||^2)
        return 0.5 * regularization * (jnp.sum(diff_y ** 2) + jnp.sum(diff_x ** 2))

    # JAX automatically computes the gradient vector (P @ c) for us!
    penalty_grad_fn = grad(roughness_penalty)

    def matvec(v):
        # A.T @ A @ v
        Av = A_sparse @ v
        AtAv = A_sparse.T @ Av

        # Add Smoothness Penalty gradient
        # This adds the term (lambda * D.T @ D) @ v
        Pv = penalty_grad_fn(v)

        # Small ridge for absolute numerical stability
        return AtAv + Pv + 1e-14 * v

    b_vector = A_sparse.T @ z
    coeffs_flat, info = cg(matvec, b_vector, maxiter=maxiter, tol=1e-10)

    return coeffs_flat.reshape(n_by, n_bx).T

#B-spline with ridge regression fitting
@partial(jit, static_argnums=(5, 6))
def _fit_solver_kernel_ridge(x, y, z, x_knots, y_knots, degree, maxiter, regularization):
    x = jnp.ravel(x)
    y = jnp.ravel(y)
    z = jnp.ravel(z)

    N = x.shape[0]
    n_bx = x_knots.shape[0] - degree - 1
    n_by = y_knots.shape[0] - degree - 1
    M = n_bx * n_by

    # 1. Local Support Lookup
    # Find which knot interval we are in. Clip ensures we don't go out of basis bounds.
    idx_x = jnp.clip(jnp.searchsorted(x_knots, x, side='right') - 1, degree, n_bx - 1)
    idx_y = jnp.clip(jnp.searchsorted(y_knots, y, side='right') - 1, degree, n_by - 1)

    def get_active_basis(val, knot_idx, knots):
        # Calculate the (degree+1) relevant basis functions
        active_indices = knot_idx - jnp.arange(degree, -1, -1)
        results = vmap(lambda k: _bspline_basis(val, k, knots, degree))(active_indices)

        # Boundary Patch:
        # Standard Cox-de Boor is half-open [a, b).
        # For the exact end-point of the domain, we must force the very last basis to 1.0.
        is_end = jnp.isclose(val, knots[-1], atol=1e-10)
        # The last basis function in the active window corresponds to index 'degree'
        end_mask = jnp.arange(degree + 1) == degree
        results = jnp.where(is_end, jnp.where(end_mask, 1.0, 0.0), results)

        return results, active_indices

    bx_vals, bx_inds = vmap(lambda v, i: get_active_basis(v, i, x_knots))(x, idx_x)
    by_vals, by_inds = vmap(lambda v, i: get_active_basis(v, i, y_knots))(y, idx_y)

    # 2. Build Sparse Matrix
    def compute_triplets(b_x, b_y, idx_x, idx_y):
        # --- FIX IS HERE ---
        # Indices: meshgrid(iy, ix) -> iy varies slow, ix varies fast.
        # Weights: outer(by, bx)    -> by varies slow, bx varies fast.
        # Previously: outer(bx, by) -> bx varies slow, by varies fast (MISMATCH)
        w_block = jnp.outer(b_y, b_x).flatten()

        iy_grid, ix_grid = jnp.meshgrid(idx_y, idx_x, indexing='ij')
        global_indices = iy_grid * n_bx + ix_grid
        return global_indices.flatten(), w_block

    all_indices, all_weights = vmap(compute_triplets)(bx_vals, by_vals, bx_inds, by_inds)

    row_indices = jnp.repeat(jnp.arange(N), (degree + 1) ** 2)
    A_sparse = sparse.BCOO(
        (all_weights.flatten(), jnp.column_stack((row_indices, all_indices.flatten()))),
        shape=(N, M)
    )

    # 3. Solve (Regularized Least Squares)
    def matvec(v):
        Av = A_sparse @ v
        AtAv = A_sparse.T @ Av
        return AtAv + regularization * v

    b_vector = A_sparse.T @ z
    coeffs_flat, info = cg(matvec, b_vector, maxiter=maxiter, tol=1e-10)

    # Reshape to (n_by, n_bx) then transpose to (n_bx, n_by) for consistency with 'ij' indexing
    return coeffs_flat.reshape(n_by, n_bx).T

def _fit_solver_kernel(x, y, z, x_knots, y_knots, degree, maxiter=500, regularization=1e-4, method='penalized'):
    if method in ['penalized', 'pen', 'Pspline', 'P-spline']:
        return _fit_solver_kernel_penalized(x, y, z, x_knots, y_knots, degree, maxiter, regularization)
    elif method in ['ridge', 'rig', 'Ridge', 'shink coeff']:
        return _fit_solver_kernel_ridge(x, y, z, x_knots, y_knots, degree, maxiter, regularization)
    else:
        raise ValueError(f'Unknown method {method} requested for bivariate spline. Choose either penalized or ridge.')
@partial(jit, static_argnums=(5,))
def _evaluate_single(x, y, coeffs, x_knots, y_knots, degree):
    # Optimized prediction using the same "active basis" logic as fit
    # This is O(degree) instead of O(N_knots)

    n_bx = x_knots.shape[0] - degree - 1
    n_by = y_knots.shape[0] - degree - 1

    # Identify active span
    idx_x = jnp.clip(jnp.searchsorted(x_knots, x, side='right') - 1, degree, n_bx - 1)
    idx_y = jnp.clip(jnp.searchsorted(y_knots, y, side='right') - 1, degree, n_by - 1)

    # Get active indices (Global indices of the basis functions)
    ix_active = idx_x - jnp.arange(degree, -1, -1)
    iy_active = idx_y - jnp.arange(degree, -1, -1)

    # Compute basis values
    bx = vmap(lambda k: _bspline_basis(x, k, x_knots, degree))(ix_active)
    by = vmap(lambda k: _bspline_basis(y, k, y_knots, degree))(iy_active)

    # Boundary fixes
    is_end_x = jnp.isclose(x, x_knots[-1], atol=1e-10)
    bx = jnp.where(is_end_x, jnp.where(jnp.arange(degree + 1) == degree, 1.0, 0.0), bx)

    is_end_y = jnp.isclose(y, y_knots[-1], atol=1e-10)
    by = jnp.where(is_end_y, jnp.where(jnp.arange(degree + 1) == degree, 1.0, 0.0), by)

    # Extract relevant coefficients
    # coeffs is (n_bx, n_by). We need coeffs[ix, iy]
    # We use vector indexing.
    # meshgrid here makes a 4x4 grid of indices to pull from the coeff matrix
    c_grid = coeffs[ix_active[:, None], iy_active]

    # Dot product: sum(C_ij * Bx_i * By_j)
    return jnp.einsum('ij,i,j->', c_grid, bx, by)


def create_clamped_knots(internal, degree=3):
    return jnp.concatenate([
        jnp.full(degree + 1, internal[0]),
        internal,
        jnp.full(degree + 1, internal[-1])
    ])


def bivariate_spline_interp(z, x_knots, y_knots, degree=3, maxiter=5000, regularization=1e-12, clamp=True):
    # 1. Validation: Ensure no repeats in the input domain (prevents singular matrix)
    if jnp.unique(x_knots).shape[0] != x_knots.shape[0]:
        raise ValueError('x_knots must be unique')
    if jnp.unique(y_knots).shape[0] != y_knots.shape[0]:
        raise ValueError('y_knots must be unique')

    # 2. Grid Construction (Training Data)
    x_grid, y_grid = jnp.meshgrid(x_knots, y_knots)  # Default indexing='xy' means x_grid varies along columns

    if z.shape != x_grid.shape:
        # Transpose z if usage implies matrix indexing (rows=y, cols=x)
        if z.shape == x_grid.T.shape:
            z = z.T
        else:
            raise ValueError(f"Shape mismatch: z is {z.shape}, grid is {x_grid.shape}")

    # 3. Knot Construction
    # If we want to clamp the spline to the edges of the data:
    if clamp:
        x_spline_knots = create_clamped_knots(x_knots, degree)
        y_spline_knots = create_clamped_knots(y_knots, degree)
    else:
        x_spline_knots = x_knots
        y_spline_knots = y_knots

    # 4. Fit
    coeffs = _fit_solver_kernel(x_grid, y_grid, z, x_spline_knots, y_spline_knots, degree, maxiter, regularization)

    # 5. Predict
    # We bake the knot vectors into the lambda so the user doesn't need to manage them
    predict_fn = vmap(lambda x, y: _evaluate_single(x, y, coeffs, x_spline_knots, y_spline_knots, degree))

    return predict_fn



def main():
    degree = 3
    x_knot = jnp.linspace(0, 3.15, 8)
    y_knot = jnp.linspace(0, 3.15, 8)

    # Generate Training Data
    x, y = jnp.meshgrid(x_knot, y_knot)
    z = jnp.sin(x) * jnp.cos(y)

    # Fit
    spline_predict = bivariate_spline_interp(z, x_knot, y_knot, degree = degree)

    # Predict
    test_x = jnp.array([1.5, 0.5])
    test_y = jnp.array([1.5, 0.5])

    # FIX: Call the returned function directly (it's not a class anymore)
    preds = spline_predict(test_x, test_y)

    exact_val = jnp.sin(test_x) * jnp.cos(test_y)

    print("\n--- Results ---")
    print(f"Predictions: {preds}")
    print(f"True Values: {exact_val}")


if __name__ == "__main__":
    main()