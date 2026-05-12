# Getting Started

This tutorial builds a finite-difference operator on a non-uniform grid and
applies it to a vector.

## Build a Grid

The main grid API returns points and quadrature weights together:

```@example getting-started
using FDGrids
using LinearAlgebra

g = grid(32, -1, 1, GaussLobattoGrid())
(length(g.xs), length(g.ws), first(g.xs), last(g.xs))
```

`g.xs` contains grid points in ascending order and `g.ws` contains quadrature
weights associated with those points.

## Construct a Differentiation Matrix

`DiffMatrix(xs, width, order)` constructs a derivative operator on the nodes
`xs`. The stencil width must be odd.

```@example getting-started
D = DiffMatrix(g.xs, 5, 1)
size(D)
```

This is a first-derivative operator with a five-point stencil. On interior rows
the stencil is centered; near the boundaries it is shifted to use one-sided
stencils of the same width.

## Apply the Operator

Use `mul!` for repeated application:

```@example getting-started
u  = sin.(g.xs)
du = similar(u)

mul!(du, D, u)
du[1:4]
```

For inspection, testing, or comparison with dense linear algebra, expand the
compact operator:

```@example getting-started
Df = full(D)
du ≈ Df * u
```

## Choose a Wider Stencil

The same grid can be used with different derivative orders and stencil widths:

```@example getting-started
D2 = DiffMatrix(g.xs, 7, 2)   # second derivative, 7-point stencil
d2u = similar(u)
mul!(d2u, D2, u)

d2u[1:4]
```

Wider stencils use more points per row and can improve accuracy for smooth
functions, but they also increase the bandwidth of the operator and the amount
of halo data needed in decomposed-domain applications.

