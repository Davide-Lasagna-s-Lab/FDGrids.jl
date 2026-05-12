# Grids and Quadrature

The grid API constructs points and quadrature weights together:

```julia
g = grid(M, l, h, distribution)
```

The return value is a named tuple with fields `xs` and `ws`. The points are in
ascending order and the weights approximate integrals on `[l,h]`:

```math
\int_l^h f(x)\,dx \approx \sum_i f(x_i) w_i.
```

## Available Distributions

### [`UniformGrid`](@ref)

`UniformGrid()` places equally spaced points on `[l,h]`. Its weights are the
composite trapezoidal rule: half weights at the endpoints and full cell widths
in the interior.

### [`MappedGrid`](@ref)

`MappedGrid(α, order)` places mapped Chebyshev-like points:

```math
x_j =
\frac{\sin^{-1}(-\alpha \cos(\pi j/(M-1)))}{\sin^{-1}(\alpha)}
\frac{h-l}{2}
+ \frac{h+l}{2},
\qquad j = 0,\ldots,M-1.
```

The clustering parameter satisfies `0 < α ≤ 1`. Smaller `α` gives stronger
endpoint clustering; `α = 1` gives uniform spacing. Quadrature weights are
computed with composite Newton-Cotes panels of degree `order`.

High-order Newton-Cotes rules on strongly non-uniform nodes may produce
negative weights. If a positive definite weighted inner product is required,
prefer `GaussLobattoGrid()`.

### [`GaussLobattoGrid`](@ref)

`GaussLobattoGrid()` uses Chebyshev-Lobatto points:

```math
x_j = \frac{l+h}{2}
    + \frac{h-l}{2}\cos\left(\frac{\pi(M-1-j)}{M-1}\right),
\qquad j = 0,\ldots,M-1.
```

The associated quadrature is Clenshaw-Curtis quadrature, with positive weights
for the grids supported by this package. This is often the best default for
weighted adjoints.

## Examples

```@example grids
using FDGrids

g = grid(64, -1, 1, GaussLobattoGrid())
sum(exp.(g.xs) .* g.ws)
```

```@example grids
g = grid(64, -1, 1, UniformGrid())
all(g.ws .> 0)
```

See the [API Reference](../api.md#Grids) for docstrings and signatures.
