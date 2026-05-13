# Adjoints

`FDGrids.jl` provides ordinary and weighted adjoints for `DiffMatrix`.

## Ordinary Adjoint

```julia
Dt = adjoint(D)
```

constructs an `AdjointDiffMatrix` representing `transpose(D)`. The adjoint
operator has its own compact coefficient storage, arranged for efficient
`mul!` application.

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

Finite-difference operators are often paired with quadrature weights. Given
positive weights `w` and `W = Diagonal(w)`, the weighted adjoint is

```math
D^+ = W^{-1} D^T W.
```

It is constructed with:

```julia
Dp = adjoint(D, w)
```

The weights are folded into the stored coefficients, so applying `Dp` does not
need to form or multiply by dense diagonal matrices.

## Weighted Identity

The weighted adjoint satisfies:

```math
(Du)^T W v = u^T W (D^+v).
```

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

The dense reference expression is:

```@example adjoints
Dp_dense = Diagonal(1 ./ w) * full(D)' * Diagonal(w)
full(Dp) ≈ Dp_dense
```

## Structural Unwrap

`adjoint(A::AdjointDiffMatrix)` returns the parent `DiffMatrix` object:

```@example adjoints
adjoint(Dp) === D
```

For an ordinary adjoint this agrees with the usual double-adjoint identity. For
a weighted adjoint it is a structural unwrap, not the Euclidean adjoint of
`W^{-1} D^T W`.

See the [API Reference](../api.md#Adjoints) for docstrings and signatures.
