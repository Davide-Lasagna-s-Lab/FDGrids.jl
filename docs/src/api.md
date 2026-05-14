# API Reference

This page collects public signatures and docstrings. For guidance and examples,
start with the manual pages; this reference is intentionally terse. Deprecated
compatibility functions are listed separately in [Deprecated API](deprecated.md).

## Grids

```@docs
AbstractGridDistribution
MappedGrid
UniformGrid
GaussLobattoGrid
grid
```

## Finite-Difference Matrices

```@docs
DiffMatrix
full
basis_vector
```

## Multiplication

```@docs
LinearAlgebra.mul!(::AbstractArray{S, N}, ::DiffMatrix{T, WIDTH}, ::AbstractArray{S, N}, ::Val{DIM}) where {T, S, N, WIDTH, DIM}
LinearAlgebra.mul!(::AbstractArray{T, N}, ::DiffMatrix{TD, WIDTH}, ::AbstractArray{T, N}, ::Val{DIM}, ::Int, ::UnitRange) where {T, TD, N, WIDTH, DIM}
LinearAlgebra.mul!(::DiffMatrix{T, WIDTH}, ::AbstractVector, ::Int) where {T, WIDTH}
```

## Adjoints

```@docs
AdjointDiffMatrix
LinearAlgebra.adjoint(::DiffMatrix)
LinearAlgebra.adjoint(::DiffMatrix{T, WIDTH}, ::AbstractVector{T}) where {T, WIDTH}
LinearAlgebra.adjoint(::AdjointDiffMatrix)
LinearAlgebra.mul!(::AbstractArray{S, N}, ::AdjointDiffMatrix{T, WIDTH}, ::AbstractArray{S, N}, ::Val{DIM}) where {T, S, N, WIDTH, DIM}
LinearAlgebra.mul!(::AbstractArray{T, N}, ::AdjointDiffMatrix{TD, WIDTH}, ::AbstractArray{T, N}, ::Val{DIM}, ::Int, ::UnitRange) where {T, TD, N, WIDTH, DIM}
```

## Linear Solves

```@docs
DiffMatrixLU
LinearAlgebra.lu!(::DiffMatrix)
LinearAlgebra.ldiv!(::FDGrids.DiffMatrixLU, ::AbstractVector)
BandedMatrixLU
LinearAlgebra.lu!(::AbstractMatrix, ::Int, ::Int)
LinearAlgebra.ldiv!(::FDGrids.BandedMatrixLU, ::AbstractVector)
LinearAlgebra.lu(::DiffMatrix)
LinearAlgebra.ldiv!(::FDGrids.DiffMatrixLULapack{T, WIDTH}, ::AbstractVector{T}) where {T, WIDTH}
```
