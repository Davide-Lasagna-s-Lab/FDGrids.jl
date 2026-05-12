# Dimension-Wise Differentiation

`DiffMatrix` is a one-dimensional operator, but it can be applied along any
dimension of an array. This is often the most important feature in PDE codes:
the storage layout can be chosen for cache locality, FFT planning, or domain
decomposition without changing the finite-difference operator.

## First Dimension

```@example dimension-wise
using FDGrids
using LinearAlgebra

nx = 48
ny = 6

g = grid(nx, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)

ys = range(0, 2π; length = ny)
u  = [sin(x) * cos(y) for x in g.xs, y in ys]
ux = similar(u)

mul!(ux, D, u, Val(1))
size(ux)
```

Each column of `u` is differentiated independently along the first dimension.

## Second Dimension

If the grid direction is stored along the second dimension, change only the
`Val` argument:

```@example dimension-wise
u2  = [sin(x) * cos(y) for y in ys, x in g.xs]
ux2 = similar(u2)

mul!(ux2, D, u2, Val(2))
size(ux2)
```

The differentiated dimension must have length `size(D, 1)`. All other
dimensions are treated as batch dimensions.

## Higher-Dimensional Arrays

The same interface works for arrays with more dimensions:

```@example dimension-wise
nz = 4
u3 = [sin(x) * cos(y) * (1 + z)
      for z in 1:nz, x in g.xs, y in ys]

ux3 = similar(u3)
mul!(ux3, D, u3, Val(2))

size(ux3)
```

Here the derivative is taken along the second dimension. The first and third
dimensions are carried along.

## Generated Kernels

The dimension-wise methods are generated for the tuple
`(array rank, differentiated dimension, stencil width)`. The generated code:

- walks the array with the differentiated index in the requested position,
- uses one-sided boundary stencils where needed,
- uses centered stencils in the interior,
- unrolls the stencil dot product using the width encoded in the `DiffMatrix`
  type.

For normal local arrays, use:

```julia
mul!(y, D, x, Val(DIM))
```

For decomposed domains, use the lower-level methods described in
[Decomposed Domains](decomposed-domains.md).

