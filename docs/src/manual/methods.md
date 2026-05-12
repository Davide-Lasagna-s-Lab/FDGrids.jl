# Numerical Methods

This page gives the mathematical background behind the main algorithms in
`FDGrids.jl`.

## Finite-Difference Weights

Finite-difference weights are computed with Fornberg's recursive algorithm for
arbitrarily spaced nodes. Given nodes

```math
x_0, x_1, \ldots, x_n
```

and an evaluation point `ξ`, the algorithm computes weights `c_{j,k}` such that

```math
f^{(k)}(\xi) \approx \sum_{j=0}^n c_{j,k} f(x_j),
\qquad k = 0,\ldots,m.
```

This is the reason `DiffMatrix` can support non-uniform grids and arbitrary
odd stencil widths.

The implementation in `utils.jl` follows the Fornberg recursion and is adapted
from Marc Avila's `nscouette` code lineage, with Julia indexing and storage
conventions.

## Stencils and Boundary Rows

For each grid point `xs[i]`, `FDGrids.jl` chooses a stencil of the requested
width:

- centered in the interior,
- shifted to the left boundary near the first point,
- shifted to the right boundary near the last point.

Every row stores the same number of coefficients. This uniform layout is
important for the generated kernels.

## Dimension-Wise Application

The generated `mul!` kernels apply the same one-dimensional matrix to every
fiber along a selected dimension:

```math
y[\ldots, i, \ldots]
=
\sum_j D_{ij} x[\ldots, j, \ldots].
```

The generated code specializes on:

- the rank of the input array,
- the differentiated dimension,
- the stencil width.

This lets the stencil dot product be unrolled while preserving the caller's
chosen memory layout.

## Quadrature

`grid` returns quadrature weights with the nodes:

```math
\int_l^h f(x)\,dx \approx \sum_i f(x_i)w_i.
```

The available rules are:

- composite trapezoidal for `UniformGrid`,
- composite Newton-Cotes for `MappedGrid`,
- Clenshaw-Curtis for `GaussLobattoGrid`.

Clenshaw-Curtis quadrature is constructed with the explicit formula described
by Waldvogel.

## Adjoints

The ordinary adjoint represents `D^T`. The weighted adjoint represents

```math
D^+ = W^{-1}D^T W,
```

where `W = Diagonal(w)`. The weighted coefficients are precomputed once at
construction time.

## Linear Solves

The compact in-place LU factorisation uses a no-pivoting banded algorithm.
Generated triangular solves specialize on the stencil width stored in the
`DiffMatrix` type. LAPACK wrappers are available for reference and comparison,
but the compact path avoids converting to LAPACK banded workspace.

## References

- B. Fornberg, "Generation of Finite Difference Formulas on Arbitrarily Spaced
  Grids", *Mathematics of Computation*, 51(184), 699-706, 1988.
- B. Fornberg, *A Practical Guide to Pseudospectral Methods*, Cambridge
  University Press, 1998.
- J. M. López, D. Feldmann, M. Rampp, A. Vela-Martín, L. Shi, and M. Avila,
  "nsCouette: A High-Performance Code for Direct Numerical Simulations of
  Turbulent Taylor-Couette Flow", *SoftwareX*, 11, 100395, 2020.
- J. Waldvogel, "Fast Construction of the Fejér and Clenshaw-Curtis Quadrature
  Rules", *BIT Numerical Mathematics*, 46, 195-202, 2006.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th edition, Johns
  Hopkins University Press, 2013.

