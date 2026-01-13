import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from functools import partial

# def main():
#     print("Hello from splines-in-jax!")

class Spline2D:
    def __init__(self, x_knots, y_knots, coefficients=None, degree=3):
        self.degree = degree
        self.x_knots = jnp.array(x_knots)
        self.y_knots = jnp.array(y_knots)
        self.coeffs = coefficients

    @staticmethod
    def _bspline_basis(x, k, knots, degree):
        """
        Recursive Cox-de Boor algorithm.
        NOTE: 'degree' must be a static integer (not a JAX array) for the
        'if degree == 0' check to work during JIT compilation.
        """
        # Base case
        if degree == 0:
            return jnp.where((x >= knots[k]) & (x < knots[k + 1]), 1.0, 0.0)

        # --- Term 1 ---
        denom1 = knots[k + degree] - knots[k]
        safe_denom1 = jnp.where(denom1 == 0, 1.0, denom1)
        term1_val = ((x - knots[k]) / safe_denom1) * Spline2D._bspline_basis(x, k, knots, degree - 1)
        # Mask if denom was 0
        term1 = jnp.where(denom1 > 0, term1_val, 0.0)

        # --- Term 2 ---
        denom2 = knots[k + degree + 1] - knots[k + 1]
        safe_denom2 = jnp.where(denom2 == 0, 1.0, denom2)
        term2_val = ((knots[k + degree + 1] - x) / safe_denom2) * Spline2D._bspline_basis(x, k + 1, knots, degree - 1)
        term2 = jnp.where(denom2 > 0, term2_val, 0.0)

        return term1 + term2

    @staticmethod
    @partial(jit, static_argnums=(5,))  # <--- CRITICAL FIX: Mark 'degree' (index 5) as static
    def _evaluate_single(x, y, coeffs, x_knots, y_knots, degree):
        """
        Evaluates the spline at a single point (x, y).
        """
        # Because degree is static, these calculations result in concrete integers,
        # which allows jnp.arange to work correctly.
        n_basis_x = len(x_knots) - degree - 1
        n_basis_y = len(y_knots) - degree - 1

        # Compute basis vector for X
        bx = vmap(lambda i: Spline2D._bspline_basis(x, i, x_knots, degree))(jnp.arange(n_basis_x))

        # Compute basis vector for Y
        by = vmap(lambda i: Spline2D._bspline_basis(y, i, y_knots, degree))(jnp.arange(n_basis_y))

        return jnp.einsum('ij,i,j->', coeffs, bx, by)


    def fit_fast(self, x, y, z, regularization=1e-6):
        """
        Optimized fit using Local Support logic.
        computes only non-zero basis functions.
        """
        x = jnp.ravel(x)
        y = jnp.ravel(y)
        z = jnp.ravel(z)

        # 1. Constants
        deg = self.degree
        n_bx = len(self.x_knots) - deg - 1
        n_by = len(self.y_knots) - deg - 1
        n_coeffs = n_bx * n_by

        # 2. Find which knot interval each point falls into
        # We subtract 1 because searchsorted returns the insertion index (right side)
        # We clamp indices to ensure they stay within valid basis range
        # (This handles points exactly on the boundary)
        idx_x = jnp.clip(jnp.searchsorted(self.x_knots, x, side='right') - 1, deg, n_bx - 1)
        idx_y = jnp.clip(jnp.searchsorted(self.y_knots, y, side='right') - 1, deg, n_by - 1)

        # 3. Compute ONLY the active basis functions (Non-Zero)
        # For a cubic spline, we only care about 'deg+1' bases ending at idx
        # We define a fixed-size window of indices: [idx-deg, ..., idx]

        # Vectorized implementation of basis evaluation for just 4 points
        def get_active_basis(val, knot_idx, knots):
            # We want to compute basis functions: k = idx - deg, ..., idx
            # This is a small, fixed size loop (size 4 for cubic)
            active_indices = knot_idx - jnp.arange(deg, -1, -1)

            # Map the standard basis function over just these 4 indices
            # Note: We pass the *scalar* value 'val', but 'active_indices' is a vector of 4
            results = vmap(lambda k: self._bspline_basis(val, k, knots, deg))(active_indices)
            return results, active_indices

        # Get active weights and their global indices for all data points
        # Shapes: (N, deg+1)
        bx_vals, bx_inds = vmap(lambda v, i: get_active_basis(v, i, self.x_knots))(x, idx_x)
        by_vals, by_inds = vmap(lambda v, i: get_active_basis(v, i, self.y_knots))(y, idx_y)

        # 4. Construct A.T @ A and A.T @ z efficiently
        # Instead of building the huge dense A (N x M), we can construct the components directly.
        # However, for JAX 'solve', we typically need the dense LHS matrix (M x M).
        # Since M (coeffs) is usually smaller than N (data), this is fine.

        # We need to map the local small outer product (4x4) to the global matrix (MxM)

        def compute_outer_products(b_x, b_y, idx_x, idx_y, z_val):
            # b_x, b_y are shape (4,) -> outer is (4, 4)
            # idx_x, idx_y are shape (4,) global indices

            w_block = jnp.outer(b_x, b_y).flatten()  # Shape (16,)

            # Calculate global flat indices for these 16 weights
            # Global Index = row * width + col
            # We need the meshgrid of indices
            # dim x: idx_x (col indices), dim y: idx_y (row indices) -- careful with reshape
            # Correct: we are mapping to a flattened coefficient vector of size (n_bx * n_by)
            # Row index in Coeff Grid is 'idx_y', Col index is 'idx_x'

            # Meshgrid of indices for the block
            iy_grid, ix_grid = jnp.meshgrid(idx_y, idx_x, indexing='ij')
            global_indices = iy_grid * n_bx + ix_grid
            global_indices = global_indices.flatten()

            # Terms for A.T @ z  -> basis_val * z
            # The contribution of this data point to the RHS vector
            rhs_contrib = w_block * z_val

            return global_indices, w_block, rhs_contrib

        # Map over all data points
        all_indices, all_weights, all_rhs = vmap(compute_outer_products)(
            bx_vals, by_vals, bx_inds, by_inds, z
        )

        # 5. Assemble Global Matrices using Scatter/Add
        # This prevents storing the massive N x M matrix.

        # LHS = A.T @ A
        # This is tricky to assemble purely from scatters without huge memory in JAX
        # because (A.T @ A) involves cross-terms between different data points.

        # --- ALTERNATIVE FAST PATH (Semi-Dense) ---
        # If N is large but M is manageable, we can just construct A sparsely?
        # JAX BCOO is good for this.

        from jax.experimental import sparse

        # Coordinates for the sparse design matrix A
        # Rows: 0..N (repeated 16 times per point)
        # Cols: all_indices
        # Data: all_weights

        N = len(x)
        M = n_coeffs

        row_indices = jnp.repeat(jnp.arange(N), (deg + 1) ** 2)
        col_indices = all_indices.flatten()
        data_values = all_weights.flatten()

        # Create Sparse Matrix A (N x M)
        A_sparse = sparse.BCOO(
            (data_values, jnp.column_stack((row_indices, col_indices))),
            shape=(N, M)
        )

        # Compute LHS = A.T @ A + reg*I
        # Sparse matrix multiplication is memory efficient
        lhs = (A_sparse.T @ A_sparse).todense()  # Result is M x M (small)

        # Add regularization
        lhs = lhs + regularization * jnp.eye(M)

        # Compute RHS = A.T @ z
        rhs = A_sparse.T @ z

        # 6. Solve
        coeffs_flat = jnp.linalg.solve(lhs, rhs)
        self.coeffs = coeffs_flat.reshape(n_bx, n_by)
        print(f"Fast Fit complete. Coeffs shape: {self.coeffs.shape}")


    def fit(self, x, y, z, regularization=1e-6):
        x = jnp.ravel(x)
        y = jnp.ravel(y)
        z = jnp.ravel(z)

        n_basis_x = len(self.x_knots) - self.degree - 1
        n_basis_y = len(self.y_knots) - self.degree - 1

        def build_row(xi, yi):
            # We must pass self.degree here. Since this helper is inside 'fit' (Python side),
            # it will unroll correctly when passed to vmap.
            bx = vmap(lambda k: self._bspline_basis(xi, k, self.x_knots, self.degree))(jnp.arange(n_basis_x))
            by = vmap(lambda k: self._bspline_basis(yi, k, self.y_knots, self.degree))(jnp.arange(n_basis_y))
            return jnp.outer(bx, by).flatten()

        A = vmap(build_row)(x, y)

        lhs = A.T @ A + regularization * jnp.eye(A.shape[1])
        rhs = A.T @ z

        coeffs_flat = jnp.linalg.solve(lhs, rhs)
        self.coeffs = coeffs_flat.reshape(n_basis_x, n_basis_y)
        print(f"Fit complete. Coeffs shape: {self.coeffs.shape}")


    def predict(self, x, y):
        if self.coeffs is None:
            raise ValueError("Spline not fitted yet.")

        # We pass self.degree (an int) to the JIT-compiled function.
        # The static_argnums=(5,) tells JAX to treat this int as a constant.
        return vmap(self._evaluate_single, in_axes=(0, 0, None, None, None, None))(
            x, y, self.coeffs, self.x_knots, self.y_knots, self.degree
        )


    def derivative(self, x, y, wrt='x'):
        if self.coeffs is None:
            raise ValueError("Spline not fitted yet.")

        eval_fn = lambda _x, _y: self._evaluate_single(
            _x, _y, self.coeffs, self.x_knots, self.y_knots, self.degree
        )

        if wrt == 'x':
            d_fn = vmap(grad(eval_fn, argnums=0))
        elif wrt == 'y':
            d_fn = vmap(grad(eval_fn, argnums=1))

        return d_fn(x, y)


def interp_3d(x, y, z):
    xx = jnp.ravel(x)
    yy = jnp.ravel(y)
    if z.shape == (xx.shape[0],yy.shape[0]):
        normal_order = True
    elif z.shape == (yy.shape[0],xx.shape[0]):
        normal_order = False
        print("Order of variables (x, y in spline interpolation) was reversed!")
    else:
        raise ValueError("Matrix z needs to have dimensions of the input vectors x and y.")
    if normal_order:
        x_grid, y_grid = jnp.meshgrid(xx, yy)
        spline = Spline2D(xx,yy)
        spline.fit(x_grid, y_grid, z, regularization=1e-12)
    else:
        x_grid, y_grid = jnp.meshgrid(xx, yy)
        spline = Spline2D(yy, xx)
        spline.fit(y_grid, x_grid, z, regularization=1e-12)
    return(spline.predict)


def generate_demo_data():
    # True function: z = sin(x) * cos(y)
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (100,), minval=0, maxval=3.14)
    y = jax.random.uniform(key, (100,), minval=0, maxval=3.14)
    z = jnp.sin(x) * jnp.cos(y)
    return x, y, z


def main():
    # 1. Prepare Data
    x_data, y_data, z_data = generate_demo_data()

    # 2. Define Knots (internal knots + padding for degree)
    # For a cubic spline (degree 3), we pad knots.
    degree = 3
    # Create internal knots
    x_grid = jnp.linspace(0, 3.15, 8)
    y_grid = jnp.linspace(0, 3.15, 8)

    # Pad knots for B-spline requirements (k repeats at ends)
    # A robust way is usually: [x_min]*degree + internal + [x_max]*degree
    x_knots = jnp.concatenate([
        jnp.full(degree, x_grid[0]),
        x_grid,
        jnp.full(degree, x_grid[-1])
    ])
    y_knots = jnp.concatenate([
        jnp.full(degree, y_grid[0]),
        y_grid,
        jnp.full(degree, y_grid[-1])
    ])

    # 3. Initialize and Fit
    spline = Spline2D(x_knots, y_knots, degree=degree)
    spline.fit(x_data, y_data, z_data)

    # 4. Predict
    test_x = jnp.array([1.5, 0.5])
    test_y = jnp.array([1.5, 0.5])
    preds = spline.predict(test_x, test_y)

    # 5. Compute Derivatives
    # dz/dx at the test points
    dz_dx = spline.derivative(test_x, test_y, wrt='x')

    # Exact derivative check: d/dx(sin(x)cos(y)) = cos(x)cos(y)
    exact_dz_dx = jnp.cos(test_x) * jnp.cos(test_y)

    print("\n--- Results ---")
    print(f"Predictions: {preds}")
    print(f"True Values: {jnp.sin(test_x) * jnp.cos(test_y)}")
    print(f"Computed dz/dx: {dz_dx}")
    print(f"Exact dz/dx:    {exact_dz_dx}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
