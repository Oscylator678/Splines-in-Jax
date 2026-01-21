import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Spline2D:
    def __init__(self, x_knots, y_knots, coefficients, degree=3):
        self.x_knots = x_knots
        self.y_knots = y_knots
        self.coeffs = coefficients
        self.degree = degree

    def __call__(self, x, y):
        """
        Evaluates the spline at points (x, y).
        Supports scalar, vector, or meshgrid inputs automatically via vmap.
        """
        # Ensure inputs are at least 1D for vmap to work
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)

        # We assume x and y are paired points. If you want meshgrid evaluation,
        # the user should pass meshgrid arrays.
        return vmap(self._evaluate_single, in_axes=(0, 0, None, None, None, None))(
            x, y, self.coeffs, self.x_knots, self.y_knots, self.degree
        )

    # --- JAX PyTree Registration (Allows passing 'self' to JIT) ---
    def tree_flatten(self):
        # Dynamic data (leaves)
        return (self.x_knots, self.y_knots, self.coeffs), (self.degree,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, degree=aux_data[0])

    # --- 1. Factories ---
    @classmethod
    def from_grid(cls, x_grid, y_grid, z_grid, degree=3, regularization=1e-14):
        """
        Factory method for exact interpolation on a grid.
        Automatically generates the correct clamped knots.
        """
        # Generate knots such that N_basis == N_data
        x_knots = cls._generate_clamped_knots(x_grid, degree)
        y_knots = cls._generate_clamped_knots(y_grid, degree)

        # Create meshgrid for the solver
        x_train, y_train = jnp.meshgrid(x_grid, y_grid, indexing='ij')

        # Solve
        coeffs = _fit_solver_kernel(
            x_train.ravel(), y_train.ravel(), z_grid.ravel(),
            x_knots, y_knots, degree, maxiter=5000, regularization=regularization
        )

        return cls(x_knots, y_knots, coeffs, degree)

    @staticmethod
    def _generate_clamped_knots(data_grid, degree):
        """Generates knots for exact interpolation: N_knots = N_data + degree + 1"""
        N = len(data_grid)
        start, end = data_grid[0], data_grid[-1]

        # Internal knots (remove start/end to avoid duplication with padding)
        # We need N - degree - 1 internal knots
        internal = jnp.linspace(start, end, N - degree + 1)[1:-1]

        return jnp.concatenate([
            jnp.full(degree, start),
            jnp.array([start]),
            internal,
            jnp.array([end]),
            jnp.full(degree, end)
        ])

    # --- 2. Core Logic ---
    @staticmethod
    def _bspline_basis(x, k, knots, degree):
        """Recursive Cox-de Boor (Standard Implementation)."""
        if degree == 0:
            return jnp.where((x >= knots[k]) & (x < knots[k + 1]), 1.0, 0.0)

        denom1 = knots[k + degree] - knots[k]
        safe_denom1 = jnp.where(denom1 == 0, 1.0, denom1)
        term1 = ((x - knots[k]) / safe_denom1) * Spline2D._bspline_basis(x, k, knots, degree - 1)
        term1 = jnp.where(denom1 > 0, term1, 0.0)

        denom2 = knots[k + degree + 1] - knots[k + 1]
        safe_denom2 = jnp.where(denom2 == 0, 1.0, denom2)
        term2 = ((knots[k + degree + 1] - x) / safe_denom2) * Spline2D._bspline_basis(x, k + 1, knots, degree - 1)
        term2 = jnp.where(denom2 > 0, term2, 0.0)

        return term1 + term2

    @staticmethod
    @partial(jit, static_argnums=(5,))
    def _evaluate_single(x, y, coeffs, x_knots, y_knots, degree):
        n_bx = len(x_knots) - degree - 1
        n_by = len(y_knots) - degree - 1

        # Compute Basis Vectors
        bx = vmap(lambda i: Spline2D._bspline_basis(x, i, x_knots, degree))(jnp.arange(n_bx))
        by = vmap(lambda i: Spline2D._bspline_basis(y, i, y_knots, degree))(jnp.arange(n_by))

        # --- Boundary Fix (Force endpoint to 1.0) ---
        # If x is at x_max, the standard basis logic returns 0.0 for the last basis.
        # We overwrite it to 1.0 to match the clamped property.

        is_end_x = jnp.isclose(x, x_knots[-1], atol=1e-10)
        bx = jnp.where(is_end_x, bx.at[-1].set(1.0), bx)

        is_end_y = jnp.isclose(y, y_knots[-1], atol=1e-10)
        by = jnp.where(is_end_y, by.at[-1].set(1.0), by)

        # Einsum: coeffs is (n_bx, n_by)
        return jnp.einsum('ij,i,j->', coeffs, bx, by)


# --- 3. The Solver Kernel (Functional & JIT) ---
@partial(jit, static_argnums=(5, 6))
def _fit_solver_kernel(x, y, z, x_knots, y_knots, degree, maxiter, regularization):
    N = x.shape[0]
    n_bx = x_knots.shape[0] - degree - 1
    n_by = y_knots.shape[0] - degree - 1
    M = n_bx * n_by

    # 1. Local Support Lookup
    idx_x = jnp.clip(jnp.searchsorted(x_knots, x, side='right') - 1, degree, n_bx - 1)
    idx_y = jnp.clip(jnp.searchsorted(y_knots, y, side='right') - 1, degree, n_by - 1)

    # 2. Compute Active Basis (With Boundary Fixes)
    def get_active_basis_masked(val, knot_idx, knots):
        # Standard recursion
        active_indices = knot_idx - jnp.arange(degree, -1, -1)
        results = vmap(lambda k: Spline2D._bspline_basis(val, k, knots, degree))(active_indices)

        # Left Boundary (Index 0 must be 1.0)
        is_start = jnp.isclose(val, knots[0], atol=1e-10)
        mask_start = (active_indices == 0)
        results = jnp.where(is_start, jnp.where(mask_start, 1.0, 0.0), results)

        # Right Boundary (Last Index must be 1.0)
        is_end = jnp.isclose(val, knots[-1], atol=1e-10)
        mask_end = (active_indices == (len(knots) - degree - 2))
        results = jnp.where(is_end, jnp.where(mask_end, 1.0, 0.0), results)

        return results, active_indices

    bx_vals, bx_inds = vmap(lambda v, i: get_active_basis_masked(v, i, x_knots))(x, idx_x)
    by_vals, by_inds = vmap(lambda v, i: get_active_basis_masked(v, i, y_knots))(y, idx_y)

    # 3. Construct Sparse Matrix Components
    def compute_triplets(b_x, b_y, idx_x, idx_y):
        w_block = jnp.outer(b_x, b_y).flatten()
        # Note: We are mapping into a flattened coefficient vector of size (n_bx * n_by)
        # We must be consistent with the reshape at the end.
        # Here we assume Row-Major (Y-inner) ordering for the solve, then Transpose later.
        iy_grid, ix_grid = jnp.meshgrid(idx_y, idx_x, indexing='ij')
        global_indices = iy_grid * n_bx + ix_grid
        return global_indices.flatten(), w_block

    all_indices, all_weights = vmap(compute_triplets)(bx_vals, by_vals, bx_inds, by_inds)

    row_indices = jnp.repeat(jnp.arange(N), (degree + 1) ** 2)
    A_sparse = sparse.BCOO(
        (all_weights.flatten(), jnp.column_stack((row_indices, all_indices.flatten()))),
        shape=(N, M)
    )

    # 4. Matrix-Free CG Solve
    def matvec(v):
        return (A_sparse.T @ (A_sparse @ v)) + regularization * v

    b_vector = A_sparse.T @ z
    coeffs_flat, _ = cg(matvec, b_vector, maxiter=maxiter, tol=1e-14)

    # 5. Reshape and Transpose
    # reshape(n_by, n_bx) -> Matches the 'iy_grid * n_bx + ix_grid' logic
    # .T -> Switches to (n_bx, n_by) so that Dim 0 is X and Dim 1 is Y for the Predictor.
    return coeffs_flat.reshape(n_by, n_bx).T


def main():
    # 1. Define Data
    x_grid = jnp.linspace(0, 3.14, 25)
    y_grid = jnp.linspace(0, 3.14, 25)
    xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')
    z_data = jnp.sin(xx) * jnp.cos(yy)

    # 2. Fit (One-liner now)
    spline = Spline2D.from_grid(x_grid, y_grid, z_data, degree=3)

    # 3. Predict (Pass random points directly)
    key = jax.random.PRNGKey(42)
    test_x = jax.random.uniform(key, (10,), minval=0, maxval=3.14)
    test_y = jax.random.uniform(key, (10,), minval=0, maxval=3.14)

    preds = spline(test_x, test_y)
    print("Predictions:", preds)


if __name__ == "__main__":
    main()