# Adjoints

`FDGrids.jl` provides compact ordinary and weighted adjoints for `DiffMatrix`.
Use this page for the public workflow. The storage layout is described in
[Internal Layout and Kernels](internals.md#Adjoint-Operator-Layout).

## Ordinary Adjoint

```julia
Dt = adjoint(D)
```

constructs an `AdjointDiffMatrix` representing `transpose(D)`.

```@example adjoints
using FDGrids
using LinearAlgebra
using Random

Random.seed!(3)

g = grid(48, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)
Dt = adjoint(D)

v = randn(length(g.xs))
y = similar(v)
mul!(y, Dt, v)

y[1:3]
```

For debugging, the dense interpretation agrees with `full(D)'`:

```@example adjoints
full(Dt) ≈ full(D)'
```

## Weighted Adjoint

With positive quadrature weights `w` and `W = Diagonal(w)`, the weighted
adjoint is

```math
D^+ = W^{-1}D^T W.
```

It is constructed with:

```julia
Dp = adjoint(D, w)
```

This is the natural adjoint for the weighted inner product defined by the grid
weights.

```@example adjoints
w  = g.ws
W  = Diagonal(w)
Dp = adjoint(D, w)

u = randn(length(g.xs))
v = randn(length(g.xs))

Du  = similar(u)
Dpv = similar(v)

mul!(Du, D, u)
mul!(Dpv, Dp, v)

(Du' * W * v) ≈ (u' * W * Dpv)
```

The weights are folded into the stored coefficients, so applying `Dp` does not
form or multiply by dense diagonal matrices.

For a dense reference:

```@example adjoints
Dp_dense = Diagonal(1 ./ w) * full(D)' * Diagonal(w)
full(Dp) ≈ Dp_dense
```

## Structural Unwrap

`adjoint(A::AdjointDiffMatrix)` returns the parent `DiffMatrix` object:

```@example adjoints
adjoint(Dp) === D
```

For an ordinary adjoint this agrees with the usual double-adjoint intuition.
For a weighted adjoint it is a structural unwrap, not the Euclidean adjoint of
`W^{-1}D^T W`.

## When To Use Weighted Adjoints

Use `adjoint(D, g.ws)` when inner products in your discretization are meant to
approximate integrals:

```math
\langle u,v\rangle_W = u^T W v.
```

For positive weights, prefer `GaussLobattoGrid()` or `UniformGrid()`. Some
`MappedGrid` Newton-Cotes weights can be negative on strongly non-uniform grids,
which makes them unsuitable for a positive definite inner product.

See [Numerical Methods](methods.md#Adjoints) for the mathematical identity and
the [API Reference](../api.md#Adjoints) for signatures.
