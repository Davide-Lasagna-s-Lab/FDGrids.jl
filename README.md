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
- Optional even/odd boundary symmetry using mirror stencils.
- Fast `mul!` application along any array dimension.
- Ordinary and quadrature-weighted adjoints.
- Grid constructors with matching quadrature weights.
- Compact banded LU factorisation and triangular solves.
- Lower-level hooks for slab-local or decomposed-domain storage with explicit
  or halo-aware ghost-cell indexing, including in-place accumulation into an
  existing output array.
- Optional CUDA extension: forward and adjoint `mul!` on NVIDIA GPUs with
  automatic dispatch — no code changes required beyond `using CUDA`.

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

Boundary rows normally use one-sided stencils. If a solution has known parity
at a boundary, pass a `(left, right)` symmetry tuple to replace those rows with
mirror stencils:

```julia
Dsym = DiffMatrix(g.xs, 5, 1;
    symmetry = (EvenSymmetry(first(g.xs)), OddSymmetry(last(g.xs))))
```

Use `NoSymmetry()` for a side that should keep its default one-sided stencil:

```julia
Dleft = DiffMatrix(g.xs, 5, 2;
    symmetry = (EvenSymmetry(first(g.xs)), NoSymmetry()))
```

The symmetry centre is explicit. For active sides it must lie at or beyond the
corresponding boundary (`c ≤ first(xs)` on the left, `c ≥ last(xs)` on the
right); missing or invalid centres are reported with the affected side.

Broadcasting keeps compact storage for operations that preserve the stencil
structure, such as combining compatible `DiffMatrix` objects or using
`Diagonal`/`UniformScaling` operands:

```julia
L = D .+ 0.1I      # diagonal shift
M = D .* I         # keep only the diagonal entries
```

In-place broadcast assignment writes into existing compact storage and is
efficient for repeated assembly:

```julia
A = similar(D)
A .= 2 .* D .- 0.1I
```

Use `full(D)` first for broadcasts that intentionally produce dense matrices,
such as `full(D) .+ rand(size(D)...)`.

The same one-dimensional operator can be applied along a chosen array axis:

```julia
ys = range(0, 2π; length = 8)
u2 = [sin(x) * cos(y) for x in g.xs, y in ys]
ux = similar(u2)

mul!(ux, D, u2, Val(1))
```

Pass `Val(true)` as the final argument to add a finite-difference application
into an existing output array:

```julia
mul!(ux, D, u2, Val(1), Val(true))
```

Weighted adjoints use the quadrature weights returned by `grid`:

```julia
Dp = adjoint(D, g.ws)
```

## GPU Support

Loading `CUDA` alongside `FDGrids` activates the optional CUDA extension.
The same `mul!` calls work unchanged on GPU arrays:

```julia
using FDGrids, CUDA, LinearAlgebra

g  = grid(256, -1, 1, GaussLobattoGrid())
D  = DiffMatrix(g.xs, 5, 1)
Dg = cu(D)                              # Float32 on device (use Adapt.adapt for Float64)

u  = CuArray(Float32.(sin.(g.xs)))
du = similar(u)
mul!(du, Dg, u)                         # dispatches to the GPU kernel automatically
```

The adjoint transfers and applies the same way:

```julia
Ag = cu(adjoint(D, g.ws))              # weighted adjoint on GPU
mul!(du, Ag, u)
```

See [GPU Support](docs/src/manual/gpu.md) for transfer options, limitations,
and launch-configuration tuning.

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
- [GPU Support](docs/src/manual/gpu.md)
- [Numerical Methods](docs/src/manual/methods.md)
- [Internal Layout and Kernels](docs/src/manual/internals.md)
- [Benchmarks](docs/src/manual/benchmarks.md)
- [API Reference](docs/src/api.md)

## Implementation Overview

`DiffMatrix` stores only the local stencil coefficients for each row. Generated
kernels use the stencil width encoded in the matrix type to apply the operator
without forming a dense matrix. The compact solve path reuses the same banded
storage for no-pivot LU factors; a LAPACK-backed pivoted path is also available
for reference and comparison.

See the manual for the detailed numerical methods, storage layouts, and
benchmarks.
