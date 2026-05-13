# Finite-Difference Operators

`DiffMatrix` is a compact representation of a one-dimensional
finite-difference differentiation matrix. It behaves like an `AbstractMatrix`,
but it stores exactly `width` coefficients per row rather than a dense `N×N`
matrix.

Use it when the same derivative operator will be applied repeatedly to one
vector or to many fibers of an array.

## Construction

```julia
D = DiffMatrix(xs, width, order)
```

- `xs` are the grid points.
- `width` is the odd stencil width.
- `order` is the derivative order.

The points may be non-uniform. They must be distinct, and their ordering must
match the arrays to which the operator will be applied.

```@example diffmatrix
using FDGrids
using LinearAlgebra

g = grid(32, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)
size(D)
```

This is a first-derivative operator with a five-point stencil. Interior rows
use centered stencils; boundary rows shift the stencil inward while preserving
the same width.

## Applying the Operator

Use `mul!` in performance-sensitive code:

```@example diffmatrix
u = sin.(g.xs)
y = similar(u)

mul!(y, D, u)
y[1:3]
```

For inspection, tests, or comparison with dense linear algebra, expand the
operator:

```@example diffmatrix
y ≈ full(D) * u
```

`full(D)` allocates a dense matrix and should not be used inside hot loops.

## Wider Stencils and Higher Derivatives

The same grid can be used with different stencil widths and derivative orders:

```@example diffmatrix
D2 = DiffMatrix(g.xs, 7, 2)
d2u = similar(u)
mul!(d2u, D2, u)

d2u[1:3]
```

Wider stencils use more points per row and can improve accuracy for smooth
functions, but they also increase bandwidth and halo requirements in
decomposed-domain applications.

## Dimension-Wise Differentiation

`DiffMatrix` is one-dimensional, but `mul!` can apply it along any dimension of
an array:

```julia
mul!(y, D, x, Val(DIM))
```

The differentiated dimension must have length `size(D, 1)`. All other
dimensions are treated as batch dimensions.

### First Dimension

```@example diffmatrix
nx = 48
ny = 6

g2 = grid(nx, -1, 1, GaussLobattoGrid())
Dx = DiffMatrix(g2.xs, 5, 1)

ys = range(0, 2π; length = ny)
u2 = [sin(x) * cos(y) for x in g2.xs, y in ys]
ux = similar(u2)

mul!(ux, Dx, u2, Val(1))
size(ux)
```

Each column is differentiated independently.

### Other Dimensions

If the grid direction is stored along another axis, change only the `Val`
argument:

```@example diffmatrix
u3 = [sin(x) * cos(y) for y in ys, x in g2.xs]
ux3 = similar(u3)

mul!(ux3, Dx, u3, Val(2))
size(ux3)
```

The same interface works for arrays with more dimensions:

```@example diffmatrix
nz = 4
u4 = [sin(x) * cos(y) * (1 + z)
      for z in 1:nz, x in g2.xs, y in ys]

ux4 = similar(u4)
mul!(ux4, Dx, u4, Val(2))

size(ux4)
```

## Storage Layout

For `N = length(xs)`, the coefficient vector has length `N * width`. Row `i`
occupies:

```math
\mathrm{coeffs}[(i-1)\,\mathrm{width}+1 : i\,\mathrm{width}].
```

The logical dense matrix is banded. Entries outside the stored stencil are
structural zeros.

Boundary rows use shifted one-sided stencils of the same width as the centered
interior stencils. This uniform row width is what allows the forward generated
kernels to be simple and fast.

See the [API Reference](../api.md#Finite-Difference-Matrices) for docstrings
and signatures.
