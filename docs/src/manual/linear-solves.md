# Linear Solves

`DiffMatrix` supports banded LU factorisation and triangular solves. The file
`src/linalg.jl` contains three layers:

1. LAPACK banded wrappers, useful for reference and comparison.
2. Generic dense-matrix banded routines, useful as algorithmic references.
3. Compact generated `DiffMatrix` routines, intended for performance-critical
   repeated solves.

## Compact Path

For performance-sensitive code, copy and factor the compact matrix in place:

```@example linear-solves
using FDGrids
using LinearAlgebra
using Random

Random.seed!(4)

g = grid(32, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 2)

Dfac = lu!(copy(D))
rhs  = randn(size(D, 1))

ldiv!(Dfac, rhs)
rhs[1:3]
```

The factorisation overwrites the compact coefficient storage. Use `copy(D)` if
the original differentiation operator is still needed.

## Reference Paths

The allocating LAPACK path is available via:

```julia
F = lu(D)
ldiv!(F, rhs)
```

The generic banded reference wrappers are:

```julia
banded_lu!(A, p, q)
banded_tril_solve!(A, b, p)
banded_triu_solve(A, b, q)
```

These operate on ordinary matrix indexing and are useful for experiments and
validation. They are not the preferred path for repeated `DiffMatrix` solves.

See the [API Reference](../api.md#Linear-Solves) for docstrings and
signatures.
