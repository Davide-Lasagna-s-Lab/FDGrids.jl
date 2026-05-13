# Grids and Quadrature

The grid API returns nodes and quadrature weights together:

```julia
g = grid(M, l, h, distribution)
```

The result is a named tuple with fields `xs` and `ws`. Nodes are ordered from
left to right, and the weights are intended to be used with those same nodes:

```math
\int_l^h f(x)\,dx \approx \sum_i f(x_i) w_i.
```

## Choosing a Distribution

| Distribution | Nodes | Weights | Typical use |
|--------------|-------|---------|-------------|
| [`UniformGrid`](@ref) | Equally spaced | Composite trapezoidal | Simple finite differences and baseline comparisons. |
| [`MappedGrid`](@ref) | Mapped Chebyshev-like | Composite Newton-Cotes | Endpoint clustering with a tunable map. |
| [`GaussLobattoGrid`](@ref) | Chebyshev-Lobatto | Clenshaw-Curtis | Positive quadrature weights and weighted adjoints. |

`UniformGrid()` is the simplest choice. It places equally spaced points on
`[l,h]` and uses half endpoint weights with full cell widths in the interior.

`MappedGrid(alpha, order)` gives endpoint clustering controlled by
`0 < alpha <= 1`. Smaller `alpha` clusters more strongly near the endpoints;
`alpha = 1` gives uniform spacing. Its quadrature weights come from composite
Newton-Cotes panels of degree `order`, and may be negative for high order or
strongly non-uniform nodes.

`GaussLobattoGrid()` uses Chebyshev-Lobatto nodes and Clenshaw-Curtis weights.
It is usually the best default when the weights define an inner product, for
example in `adjoint(D, g.ws)`.

## Basic Use

```@example grids
using FDGrids

g = grid(32, -1, 1, GaussLobattoGrid())
(length(g.xs), length(g.ws), first(g.xs), last(g.xs))
```

Use the paired weights for integration:

```@example grids
g = grid(64, -1, 1, GaussLobattoGrid())
sum(exp.(g.xs) .* g.ws)
```

Check positivity when the weights will define an inner product:

```@example grids
all(g.ws .> 0)
```

For the quadrature formulas and exactness conditions, see
[Numerical Methods](methods.md#Quadrature). For signatures, see the
[API Reference](../api.md#Grids).
