# Weighted Adjoints

Finite-difference operators are often used with quadrature weights. If
`W = Diagonal(w)`, the weighted adjoint of a matrix `D` is the operator `D⁺`
that satisfies

```math
(Du)^T W v = u^T W (D^+ v).
```

Equivalently,

```math
D^+ = W^{-1} D^T W.
```

`FDGrids.jl` constructs this operator without explicitly forming the dense
diagonal matrices.

## Verify the Identity

```@example weighted-adjoints
using FDGrids
using LinearAlgebra
using Random

Random.seed!(1)

g = grid(48, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)

w  = g.ws
W  = Diagonal(w)
Dp = adjoint(D, w)

u = randn(length(g.xs))
v = randn(length(g.xs))

Du  = similar(u)
Dpv = similar(v)

mul!(Du, D, u)
mul!(Dpv, Dp, v)

left  = Du' * W * v
right = u'  * W * Dpv

left ≈ right
```

The weighted adjoint stores the factors `w[i] / w[j]` inside its coefficient
array. This keeps subsequent `mul!` calls allocation-free and avoids applying
`W` and `W^{-1}` every time.

## Dense Comparison

For testing and debugging, compare against the dense expression:

```@example weighted-adjoints
Dp_dense = Diagonal(1 ./ w) * full(D)' * Diagonal(w)
full(Dp) ≈ Dp_dense
```

## Structural Double-Adjoint

`adjoint(A::AdjointDiffMatrix)` returns the parent `DiffMatrix` object:

```@example weighted-adjoints
adjoint(Dp) === D
```

For an unweighted adjoint this matches the usual double-adjoint intuition. For
a weighted adjoint it is a structural unwrap of the original operator, not the
Euclidean adjoint of `W^{-1} D^T W`.

