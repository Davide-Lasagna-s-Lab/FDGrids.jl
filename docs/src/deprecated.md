# Deprecated API

These functions are retained for compatibility with older `FDGrids.jl` code.
New code should use [`grid`](@ref), [`DiffMatrix`](@ref), and the standard
`LinearAlgebra` methods documented in the main [API Reference](api.md).

## Grid Points

`gridpoints` returns only grid points. Prefer `grid(...).xs`, which keeps points
and weights paired.

```@docs
gridpoints
```

## Quadrature Weights

`quadweights` and `_quadweights` are older quadrature entry points. Prefer
`grid(...).ws`, because the weight rule should match the grid distribution.

```@docs
quadweights
_quadweights
```

