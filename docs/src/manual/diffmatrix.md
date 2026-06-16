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
width. When the field has a known parity at a wall, those boundary rows can be
replaced with even/odd mirror stencils via the `symmetry` keyword — see
[Boundary Symmetry](symmetry.md).

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

## Compact Broadcasting

Broadcasting with a `DiffMatrix` keeps the result in compact stencil storage
only when the operation cannot create nonzeros outside the stored bands. The
supported matrix-shaped operands are:

- another compatible `DiffMatrix` of the same size,
- a `Diagonal` matrix,
- a `UniformScaling`, such as `I` or `3I`.

```@example diffmatrix
D3 = DiffMatrix(g.xs, 3, 1)
D5 = DiffMatrix(g.xs, 5, 2)

L = D5 .+ 0.1I
M = D3 .+ D5
N = D5 .* I

(typeof(L), typeof(M), typeof(N))
```

`D .+ 3I` adds to the diagonal. `D .* I` is the elementwise product with the
identity and therefore keeps only the diagonal entries of `D`. Diagonal matrices
behave similarly:

```@example diffmatrix
C = Diagonal(range(1.0, 2.0; length=size(D, 1)))
FDGrids.full(D .+ C) ≈ FDGrids.full(D) + C
```

Scalar multiplication is also compact:

```@example diffmatrix
2 .* D isa DiffMatrix
```

In-place broadcast assignment writes directly into an existing `DiffMatrix`
destination and is the preferred form in hot loops:

```@example diffmatrix
A = similar(D)
A .= D .+ 3I

FDGrids.full(A) ≈ FDGrids.full(D) + 3I
```

The same path is used for fused expressions such as `A .= 2 .* D .- 0.1I` and
does not allocate a temporary `DiffMatrix`.

Broadcasts that would create dense structure are deliberately unsupported as
compact `DiffMatrix` operations. For example, `D .+ 1` or `D .+ rand(size(D)...)`
would fill structural zeros outside the stencil. Use `full(D)` first when a
dense result is intended:

```@example diffmatrix
dense_result = full(D) .+ rand(size(D)...)
size(dense_result)
```

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

## Accumulating Into an Existing Output

By default, `mul!` overwrites `y`. Pass `Val(true)` as the final argument to add
the differentiated values into the current contents of `y`:

```@example diffmatrix
uy = fill(1.0, size(u2))
mul!(uy, Dx, u2, Val(1), Val(true))

uy ≈ ones(size(u2)) .+ ux
```

This is useful when assembling sums of finite-difference terms, such as a
Laplacian made from second derivatives along several array dimensions.

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
