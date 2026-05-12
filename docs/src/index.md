# FDGrids.jl

```@raw html
<p align="center">
  <img src="assets/logo.svg" alt="FDGrids.jl logo" width="640">
</p>
```

`FDGrids.jl` provides compact finite-difference differentiation matrices on
uniform and non-uniform grids. The same one-dimensional operator can be applied
to vectors, matrices, and higher-dimensional arrays along any selected
dimension.

The package is useful when a PDE, stability, or time-stepping code repeatedly
applies the same one-dimensional finite-difference operator to many fibers of an
array. It stores only the local stencil coefficients and supplies generated
`mul!` kernels for the common application paths.

## Highlights

- Finite-difference coefficients on arbitrary non-uniform nodes.
- User-chosen odd stencil widths.
- Dimension-wise `mul!` along any array axis.
- Ordinary and weighted adjoint operators.
- Grid constructors with matching quadrature weights.
- Compact in-place LU factorisation and generated triangular solves.
- Lower-level hooks for slab-local and decomposed-domain arrays.

## Quick Example

```@example quickstart
using FDGrids
using LinearAlgebra

g = grid(64, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)

u  = sin.(g.xs)
du = similar(u)
mul!(du, D, u)

du[1:3]
```

`DiffMatrix` behaves like an `AbstractMatrix`, but the dense matrix is only
formed when requested:

```@example quickstart
M = full(D)
du ≈ M * u
```

## Documentation Map

The tutorials are the best place to start:

- [Getting Started](tutorials/getting-started.md)
- [Dimension-Wise Differentiation](tutorials/dimension-wise.md)
- [Weighted Adjoints](tutorials/weighted-adjoints.md)
- [Decomposed Domains](tutorials/decomposed-domains.md)

The manual pages give a more detailed account of the implementation and
mathematics:

- [Grids and Quadrature](manual/grids.md)
- [Finite-Difference Operators](manual/diffmatrix.md)
- [Adjoints](manual/adjoints.md)
- [Linear Solves](manual/linear-solves.md)
- [Numerical Methods](manual/methods.md)
- [Internal Layout and Kernels](manual/internals.md)

For function signatures and docstrings, see the [API Reference](api.md).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl")
```

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
