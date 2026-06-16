# FDGrids.jl

```@raw html
<p align="center">
  <img src="assets/logo.svg" alt="FDGrids.jl logo" width="640">
</p>
```

`FDGrids.jl` builds compact one-dimensional finite-difference operators on
uniform and non-uniform grids. A single `DiffMatrix` can be applied to a vector
or along any selected dimension of a higher-dimensional array.

The package is aimed at solvers that repeatedly apply the same
finite-difference operator to many array fibers. It stores only local stencil
coefficients and uses generated kernels for the common multiplication and
triangular-solve paths.

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

`DiffMatrix` behaves like an `AbstractMatrix`, but the dense matrix is formed
only when requested:

```@example quickstart
du ≈ full(D) * u
```

## What To Read

| Page | Use it for |
|------|------------|
| [Scope and Limitations](manual/scope.md) | What the package does and deliberately does not do. |
| [Grids and Quadrature](manual/grids.md) | Choosing grid distributions and paired quadrature weights. |
| [Finite-Difference Operators](manual/diffmatrix.md) | Constructing and applying `DiffMatrix` objects. |
| [Boundary Symmetry](manual/symmetry.md) | Even/odd mirror stencils for boundary rows. |
| [Adjoints](manual/adjoints.md) | Ordinary and weighted adjoints. |
| [Linear Solves](manual/linear-solves.md) | Compact banded LU, pivoting, and boundary value problems. |
| [Decomposed Domains](manual/decomposed-domains.md) | Applying operators to slab-local arrays with halos. |
| [GPU Support](manual/gpu.md) | Running forward and adjoint `mul!` on NVIDIA GPUs. |
| [Numerical Methods](manual/methods.md) | The algorithms and mathematical assumptions. |
| [Internal Layout and Kernels](manual/internals.md) | Storage layouts, generated kernels, and implementation invariants. |
| [Benchmarks](manual/benchmarks.md) | Performance methodology and results. |

For function signatures and docstrings, see the [API Reference](api.md).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl")
```
