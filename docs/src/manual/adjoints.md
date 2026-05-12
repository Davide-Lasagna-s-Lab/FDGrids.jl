# Adjoints

`FDGrids.jl` provides ordinary and weighted adjoints for `DiffMatrix`.

## Ordinary Adjoint

```julia
Dt = adjoint(D)
```

constructs an `AdjointDiffMatrix` representing `transpose(D)`. The adjoint
operator has its own compact coefficient storage, arranged for efficient
`mul!` application.

## Weighted Adjoint

Given positive weights `w` and `W = Diagonal(w)`, the weighted adjoint is

```math
D^+ = W^{-1} D^T W.
```

It is constructed with:

```julia
Dp = adjoint(D, w)
```

The weights are folded into the stored coefficients, so applying `Dp` does not
need to form or multiply by dense diagonal matrices.

## Identity

The weighted adjoint satisfies:

```math
(Du)^T W v = u^T W (D^+v).
```

```@example adjoints
using FDGrids
using LinearAlgebra
using Random

Random.seed!(3)

g = grid(48, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)

w  = g.ws
W  = Diagonal(w)
Dp = adjoint(D, w)

u = randn(length(g.xs))
v = randn(length(g.xs))

Du = similar(u)
Pv = similar(v)

mul!(Du, D, u)
mul!(Pv, Dp, v)

(Du' * W * v) ≈ (u' * W * Pv)
```

## Structural Unwrap

`adjoint(A::AdjointDiffMatrix)` returns the parent `DiffMatrix` object. For an
ordinary adjoint this agrees with the usual double-adjoint identity. For a
weighted adjoint this is a structural unwrap, not the Euclidean adjoint of the
weighted operator.

See the [API Reference](../api.md#Adjoints) for docstrings and signatures.
