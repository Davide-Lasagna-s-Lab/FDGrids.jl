# Decomposed Domains

The ordinary dimension-wise interface

```julia
mul!(y, D, x, Val(DIM))
```

assumes that `x` and `y` contain the full differentiated dimension. For
distributed arrays or slab-local storage, `FDGrids.jl` also provides a lower
level interface:

```julia
mul!(y_local, D, x_local, Val(DIM), global_idx, local_rng)
```

This applies the global operator `D` to a local array.

## Arguments

- `DIM` is the local array dimension being differentiated.
- `global_idx` is the global row index corresponding to local index `1`.
- `local_rng` selects the local rows to compute along dimension `DIM`.

The method does not communicate halo values. A domain-decomposition layer must
ensure that `x_local` contains all values required by the stencils in
`local_rng`.

## Slab with Halo Points

Suppose the global differentiated direction has length `64`, the stencil width
is `5`, and a rank owns global rows `17:32`. A centered width-5 stencil needs
two neighboring values on either side, so the local storage includes rows
`15:34`.

```@example decomposed
using FDGrids
using LinearAlgebra
using Random

Random.seed!(2)

nx = 64
ny = 8

g = grid(nx, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)

owned  = 17:32
stored = 15:34
local_owned = (first(owned) - first(stored) + 1):(last(owned) - first(stored) + 1)

x_local = randn(length(stored), ny)
y_local = similar(x_local)

mul!(y_local, D, x_local, Val(1), first(stored), local_owned)

size(y_local)
```

Only the rows in `local_owned` are meaningful outputs from this call. The halo
rows in `y_local` are just storage.

## Decomposed Direction on Another Axis

The decomposed direction does not need to be the first local array dimension:

```@example decomposed
x_local_2 = randn(ny, length(stored))
y_local_2 = similar(x_local_2)

mul!(y_local_2, D, x_local_2, Val(2), first(stored), local_owned)

size(y_local_2)
```

This is the same global operation, but with the differentiated index stored
along local dimension `2`.

## Boundary Workflows

If a local array does not contain enough halo values, restrict `local_rng` to
the rows whose stencil support is local. Boundary or overlap rows can then be
filled by another communication/computation step.

The interface deliberately leaves communication policy outside `FDGrids.jl`.
That keeps the finite-difference kernels reusable across threaded, MPI, GPU, or
custom slab-storage implementations.

