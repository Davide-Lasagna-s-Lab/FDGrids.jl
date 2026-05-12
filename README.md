<p align="center">
  <img src="assets/logo.svg" alt="FDGrids.jl logo" width="640">
</p>

# FDGrids.jl

Finite-difference differentiation matrices and one-dimensional grids for Julia.

`FDGrids.jl` is a small package for building compact finite-difference operators
on uniform and non-uniform grids. It provides:

- finite-difference matrices on arbitrary one-dimensional nodes,
- finite-difference coefficients for arbitrary-width stencils on non-uniform grids,
- fast `mul!` application along any dimension of an array,
- adjoint and weighted-adjoint finite-difference operators,
- grid constructors with matching quadrature weights,
- compact LU factorisation and triangular solves for `DiffMatrix`.

The package is aimed at numerical PDE, stability, and spectral/finite-difference
workflows where the same one-dimensional differentiation operator is applied
many times to vectors, matrices, or higher-dimensional arrays.

## Installation

From the Julia package prompt:

```julia
pkg> add https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl
```

or from Julia:

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl")
```

## Quick Start

```julia
using FDGrids
using LinearAlgebra

# Grid points and quadrature weights on [-1, 1]
g = grid(64, -1, 1, GaussLobattoGrid())

# First derivative with a 5-point stencil
D = DiffMatrix(g.xs, 5, 1)

u  = sin.(g.xs)
du = similar(u)
mul!(du, D, u)
```

`DiffMatrix` behaves like an `AbstractMatrix`, but it stores only the local
stencil coefficients:

```julia
M = full(D)             # dense matrix, useful for inspection/tests
du_dense = M * u
du ≈ du_dense
```

For repeated application, prefer `mul!` over forming `full(D)`.

## Dimension-Wise Differentiation

The same one-dimensional operator can be applied along any dimension of an
array. If `A` is an `N`-dimensional array and `DIM` is the differentiated
dimension, then every one-dimensional fiber

```julia
A[i1, ..., :, ..., iN]
```

is differentiated independently with the same `DiffMatrix`. The differentiated
dimension must have length `size(D, 1)`; all other dimensions are carried along
as batch dimensions.

```julia
nx = 64
ny = 8

g = grid(nx, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)

u  = [sin(x) * cos(y) for x in g.xs, y in range(0, 2π; length = ny)]
ux = similar(u)

mul!(ux, D, u, Val(1))  # differentiate along the first dimension
```

For example, if the grid direction is stored along the second array dimension,
the same operator can be applied with `Val(2)`:

```julia
u  = [sin(x) * cos(y) for y in range(0, 2π; length = ny), x in g.xs]
ux = similar(u)

mul!(ux, D, u, Val(2))
```

This is useful in PDE codes where memory layout is chosen for cache locality,
parallel decomposition, or compatibility with FFT plans. The finite-difference
operator is not tied to a particular array axis.

Internally, the `mul!` methods are generated for the tuple
`(array dimension, stencil width, array rank)`. The generated loop nests keep
the differentiated index in the requested position and unroll the stencil
dot-product using the width encoded in the `DiffMatrix` type. Boundary fibers
use the stored one-sided stencils, while interior fibers use the centered
stencil block.

There are also lower-level `mul!` methods with `(global_idx, local_rng)`
arguments. Those are intended for distributed or slab-local arrays: `local_rng`
selects the local part of the differentiated dimension, while `global_idx`
states where that local range begins in the global row numbering of the
operator.

There is also a point-evaluation method when only one row of the operator is
needed:

```julia
du_at_10 = mul!(D, sin.(g.xs), 10)
```

## Grids and Quadrature

The main grid API returns points and quadrature weights together:

```julia
g = grid(64, -1, 1, GaussLobattoGrid())
integral = sum(exp.(g.xs) .* g.ws)
```

Available grid distributions are:

- `UniformGrid()`:
  equally spaced points with composite trapezoidal weights.
- `MappedGrid(α=0.5, order=4)`:
  mapped Chebyshev-like points with composite Newton-Cotes weights.
- `GaussLobattoGrid()`:
  Chebyshev-Lobatto points with Clenshaw-Curtis weights.

`GaussLobattoGrid()` is a good default when positive quadrature weights are
important, for example when constructing weighted adjoints.

The old split APIs `gridpoints`, `quadweights`, and `_quadweights` are retained
for compatibility but deprecated. New code should use `grid(...).xs` and
`grid(...).ws`.

## Adjoint Operators

The ordinary adjoint is built once and then applied with the same `mul!`
interface:

```julia
Dt = adjoint(D)

v  = cos.(g.xs)
dv = similar(v)
mul!(dv, Dt, v)
```

Weighted adjoints are also supported. If `W = Diagonal(w)`, then

```julia
Dp = adjoint(D, w)
```

represents

```julia
inv(W) * transpose(D) * W
```

without explicitly forming the dense matrices:

```julia
w  = g.ws
W  = Diagonal(w)
Dp = adjoint(D, w)

u = randn(length(g.xs))
v = randn(length(g.xs))

Du  = similar(u)
Dpv = similar(v)

mul!(Du,  D,  u)
mul!(Dpv, Dp, v)

# Weighted adjoint identity:
#     <D*u, v>_W = <u, Dp*v>_W
left  = Du' * W * v
right = u'  * W * Dpv

left ≈ right
```

Note that `adjoint(A::AdjointDiffMatrix)` returns the parent `DiffMatrix`
structurally. For a weighted adjoint, this is an unwrap of the stored parent, not
the Euclidean adjoint of the weighted operator.

## Linear Solves

`DiffMatrix` supports compact in-place LU factorisation and triangular solves:

```julia
Dfac = lu!(copy(D))
rhs  = randn(size(D, 1))
ldiv!(Dfac, rhs)
```

The linear algebra implementation has three layers:

1. LAPACK banded wrappers, useful for reference and comparison.
2. Generic dense-matrix banded LU/solve routines, useful as algorithmic
   references.
3. Generated compact `DiffMatrix` kernels, intended for performance-critical
   repeated solves.

Performance-sensitive code should use the compact path, as in the example
above.

## Numerical Methods

Finite-difference coefficients are generated using Fornberg's recursive method
for arbitrarily spaced nodes. This makes `DiffMatrix` suitable for non-uniform
grids as well as uniform grids.

The stencil width is chosen by the user:

```julia
D1 = DiffMatrix(xs, 3, 1)   # first derivative, 3-point stencil
D2 = DiffMatrix(xs, 7, 2)   # second derivative, 7-point stencil
```

Any odd stencil width is allowed, provided it is larger than the derivative
order and no wider than the number of grid points. The grid points `xs` may be
non-uniform; the finite-difference weights are computed locally for each
stencil.

The coefficient routine follows the recursive weights algorithm described by
Fornberg. The implementation in `utils.jl` is adapted from Marc Avila's
`nscouette` code lineage, with Julia indexing and storage conventions. Given
nodes `x`, an evaluation point `ξ`, and derivative order `m`, the routine builds
weights for derivatives `0:m`; `DiffMatrix` then extracts the requested
derivative column for each grid point.

Boundary rows use one-sided stencils of the same width as the interior centered
stencils. This gives a uniform compact storage layout: each row stores exactly
`width` coefficients.

For a stencil width `width`, the `DiffMatrix` storage is a flat vector of length
`length(xs) * width`. Row `i` occupies the slice

```julia
coeffs[(i - 1) * width + 1 : i * width]
```

The logical matrix is square and banded, but values outside the stored stencil
are structural zeros. `full(D)` expands this representation for testing and
inspection; `mul!` applies it directly from compact storage.

Quadrature rules are tied to the grid constructor:

- `UniformGrid` uses the composite trapezoidal rule.
- `MappedGrid` uses composite Newton-Cotes panels of configurable order.
- `GaussLobattoGrid` uses Clenshaw-Curtis quadrature.

The adjoint operator stores coefficients in output-major order. Generated
`mul!` kernels use the stencil width encoded in the matrix type to unroll the
inner loops.

Weighted adjoints are formed by folding the quadrature weights into the stored
coefficients at construction time. This avoids applying diagonal weight matrices
during every `mul!` call.

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
