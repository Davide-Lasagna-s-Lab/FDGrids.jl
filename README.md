<p align="center">
  <img src="docs/src/assets/logo.svg" alt="FDGrids.jl logo" width="640">
</p>

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Davide-Lasagna-s-Lab.github.io/FDGrids.jl/dev/)

# FDGrids.jl

`FDGrids.jl` builds compact finite-difference differentiation matrices on
uniform and non-uniform one-dimensional grids. The same operator can be applied
to vectors or along any selected dimension of a higher-dimensional array.

The package is aimed at numerical PDE, stability, and time-stepping workflows
where a fixed one-dimensional derivative operator is applied many times.

## Features

- Finite-difference operators on arbitrary one-dimensional nodes.
- User-selected odd stencil widths and derivative orders.
- Fast `mul!` application along any array dimension.
- Ordinary and quadrature-weighted adjoints.
- Grid constructors with matching quadrature weights.
- Compact banded LU factorisation and triangular solves.
- Lower-level hooks for slab-local or decomposed-domain storage.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl")
```

## Example

```julia
using FDGrids
using LinearAlgebra

g = grid(64, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)

u  = sin.(g.xs)
du = similar(u)
mul!(du, D, u)
```

`DiffMatrix` behaves like an `AbstractMatrix`, but dense expansion is meant for
inspection and tests:

```julia
du ≈ full(D) * u
```

The same one-dimensional operator can be applied along a chosen array axis:

```julia
ys = range(0, 2π; length = 8)
u2 = [sin(x) * cos(y) for x in g.xs, y in ys]
ux = similar(u2)

mul!(ux, D, u2, Val(1))
```

Weighted adjoints use the quadrature weights returned by `grid`:

```julia
Dp = adjoint(D, g.ws)
```

## Documentation

The full documentation is available at the
[development documentation site](https://Davide-Lasagna-s-Lab.github.io/FDGrids.jl/dev/).

Useful entry points:

- [Scope and Limitations](docs/src/manual/scope.md)
- [Grids and Quadrature](docs/src/manual/grids.md)
- [Finite-Difference Operators](docs/src/manual/diffmatrix.md)
- [Adjoints](docs/src/manual/adjoints.md)
- [Linear Solves](docs/src/manual/linear-solves.md)
- [Decomposed Domains](docs/src/manual/decomposed-domains.md)
- [Numerical Methods](docs/src/manual/methods.md)
- [Internal Layout and Kernels](docs/src/manual/internals.md)
- [API Reference](docs/src/api.md)

## Implementation Overview

`DiffMatrix` stores only the local stencil coefficients for each row. Generated
kernels use the stencil width encoded in the matrix type to apply the operator
without forming a dense matrix. The compact solve path reuses the same banded
storage for no-pivot LU factors; a LAPACK-backed pivoted path is also available
for reference and comparison.

See the manual for the detailed numerical methods, storage layouts, and
benchmarks.
