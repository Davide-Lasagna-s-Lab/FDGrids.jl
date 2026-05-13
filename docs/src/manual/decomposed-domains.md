# Decomposed Domains

The normal call

```julia
mul!(y, D, x, Val(DIM))
```

assumes that `x` and `y` contain the full differentiated dimension. If a solver
stores only a slab of the global domain plus halo points, use the lower-level
call:

```julia
mul!(y_local, D, x_local, Val(DIM), global_idx, local_rng)
```

This page documents only the indexing contract. Communication and halo exchange
remain the responsibility of the caller.

## Contract

- `DIM` is the local array dimension corresponding to the global grid of `D`.
- `global_idx` is the global row index represented by local index `1`.
- `local_rng` selects the local rows to compute.
- `x_local` must already contain every value touched by the stencils for
  `local_rng`.

Only entries selected by `local_rng` are meaningful outputs. Other entries of
`y_local` are left to the caller's storage convention.

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

The call computes the global rows `17:32` using storage whose local index `1`
corresponds to global row `15`.

## Decomposed Direction on Another Axis

The decomposed direction does not need to be local dimension `1`:

```@example decomposed
x_local_2 = randn(ny, length(stored))
y_local_2 = similar(x_local_2)

mul!(y_local_2, D, x_local_2, Val(2), first(stored), local_owned)

size(y_local_2)
```

This is the same global operation, but the differentiated index is stored along
local dimension `2`.

## Boundary and Halo Strategy

If a slab does not contain enough halo values for every owned row, restrict
`local_rng` to the rows whose stencil support is available. Boundary or overlap
rows can then be computed after communication, copied from neighbors, or handled
by another layer of the solver.

`FDGrids.jl` deliberately does not prescribe whether that layer is MPI,
threaded shared memory, GPU exchange, or custom slab storage. The only
requirement is that `x_local`, `global_idx`, and `local_rng` satisfy the
contract above.

For the pointer arithmetic used by this method, see
[Internal Layout and Kernels](internals.md#Decomposed-Domain-Pointers).
