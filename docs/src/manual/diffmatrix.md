# Finite-Difference Operators

`DiffMatrix` is a compact representation of a finite-difference differentiation
matrix. It behaves like an `AbstractMatrix`, but it stores exactly `width`
coefficients per row rather than a dense `N×N` matrix.

## Construction

```julia
D = DiffMatrix(xs, width, order)
```

- `xs` are the grid points.
- `width` is the odd stencil width.
- `order` is the derivative order.

The points may be non-uniform. They must be distinct, and the ordering must
match the arrays to which the operator will be applied.

## Storage Layout

For `N = length(xs)`, the coefficient vector has length `N * width`. Row `i`
occupies:

```math
\mathrm{coeffs}[(i-1)\,\mathrm{width}+1 : i\,\mathrm{width}].
```

The logical dense matrix is banded. Entries outside the stored stencil are
structural zeros.

Boundary rows use shifted one-sided stencils of the same width as the centered
interior stencils. This uniform row width is what allows the compact generated
kernels to be simple and fast.

## Applying the Operator

```@example diffmatrix
using FDGrids
using LinearAlgebra

g = grid(32, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)

u = sin.(g.xs)
y = similar(u)
mul!(y, D, u)

y[1:3]
```

For dense comparison:

```@example diffmatrix
y ≈ full(D) * u
```

Use `full(D)` for inspection and tests. Use `mul!` in performance-sensitive
loops.

## Arbitrary Array Dimensions

`mul!(y, D, x, Val(DIM))` applies the same one-dimensional operator along
dimension `DIM` of `x`. See [Dimension-Wise Differentiation](@ref) for a full
tutorial.

See the [API Reference](../api.md#Finite-Difference-Matrices) for docstrings
and signatures.
