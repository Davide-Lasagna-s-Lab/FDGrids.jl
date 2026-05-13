# Finite-Difference Operators

`DiffMatrix` represents a one-dimensional finite-difference differentiation
matrix in compact form. It behaves like an `AbstractMatrix`, but performance
critical code should apply it with `mul!` rather than forming a dense matrix.

## Construction

```julia
D = DiffMatrix(xs, width, order)
```

- `xs`: grid points, ordered as they appear in the arrays being differentiated.
- `width`: odd stencil width.
- `order`: derivative order.

```@example diffmatrix
using FDGrids
using LinearAlgebra

g = grid(32, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)
size(D)
```

The example constructs a first-derivative operator with a five-point stencil.
Interior rows are centered; boundary rows use one-sided stencils of the same
width.

## Applying to Vectors

```@example diffmatrix
u = sin.(g.xs)
du = similar(u)

mul!(du, D, u)
du[1:3]
```

For debugging or tests, compare against the dense interpretation:

```@example diffmatrix
du ≈ full(D) * u
```

`full(D)` allocates and is not intended for hot loops.

## Wider Stencils and Higher Derivatives

The same nodes can be reused with a different stencil width or derivative
order:

```@example diffmatrix
D2 = DiffMatrix(g.xs, 7, 2)
d2u = similar(u)
mul!(d2u, D2, u)

d2u[1:3]
```

Wider stencils use more points per row. They can improve accuracy for smooth
functions, but they also increase bandwidth and halo requirements.

## Applying Along an Array Dimension

`mul!(y, D, x, Val(DIM))` applies the same one-dimensional operator along
dimension `DIM`; all other dimensions are batch dimensions.

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

If the differentiated coordinate is stored along another axis, change `Val`:

```@example diffmatrix
u3 = [sin(x) * cos(y) for y in ys, x in g2.xs]
ux3 = similar(u3)

mul!(ux3, Dx, u3, Val(2))
size(ux3)
```

The differentiated dimension must have length `size(D, 1)`.

## Mutating Entries

`DiffMatrix` has a fixed compact band layout. Assignments inside the stored
stencil for a row mutate that row; assignments outside the stored stencil are
ignored because there is no storage location for them. This matters when
overwriting rows for boundary conditions. See
[Linear Solves](linear-solves.md#Boundary-Value-Problem) for a complete
example.

For storage formulas and generated-kernel details, see
[Internal Layout and Kernels](internals.md). For signatures, see the
[API Reference](../api.md#Finite-Difference-Matrices).
