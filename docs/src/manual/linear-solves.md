# Linear Solves

`DiffMatrix` can also be used as a compact banded system matrix. This is useful
after boundary rows have been overwritten, for example in collocation or
finite-difference boundary value problems.

The solve code in `src/linalg.jl` contains three paths. They share the same
banded-LU idea, but differ in storage, pivoting, and how much of the work is
specialized for `DiffMatrix`.

## Solve Paths

| Path | Entry points | Purpose |
|------|--------------|---------|
| LAPACK banded | `lu(D)`, `ldiv!(F, rhs)` | Pivoted reference path using LAPACK `gbtrf!`/`gbtrs!`. |
| Generic banded | `banded_lu!`, `banded_tril_solve!`, `banded_triu_solve` | Algorithmic reference implementation using ordinary matrix indexing. |
| Compact `DiffMatrix` | `lu!(D)`, `ldiv!(D, rhs)` | Performance path that keeps the LU factors in compact `DiffMatrix` storage. |

### LAPACK Banded Path

`lu(D::DiffMatrix)` expands the compact coefficients into LAPACK's general
banded workspace and calls `LinearAlgebra.LAPACK.gbtrf!`. The returned
`DiffMatrixLU` stores the LAPACK factor array and the pivot vector. Solves then
call `gbtrs!`.

This path is useful when pivoting is desired, or when comparing against a
well-tested library implementation. It allocates the LAPACK banded workspace
and converts out of the compact `DiffMatrix` layout, so it is not the preferred
path for repeated solves.

### Pivoting

The LAPACK path uses pivoted banded LU. In LAPACK terminology, `gbtrf!`
performs Gaussian elimination with partial pivoting restricted to the banded
storage format and records the row interchanges in a pivot vector. This makes it
the safest path when the pivot quality is unknown.

The generic and compact `DiffMatrix` paths do **not** pivot. They assume that
the pivots encountered by the banded elimination are nonzero and numerically
acceptable. This is often true for the well-conditioned model operators used in
the tests and examples after boundary rows are imposed, but it is not a general
guarantee for every discretized operator.

The reason is structural: pivoting changes the row order and may introduce fill
that no longer fits the simple fixed-row `DiffMatrix` coefficient layout. LAPACK
handles this with its own banded workspace and pivot vector. The compact path
keeps the factors inside the original `DiffMatrix` storage, which is why it is
fast, but also why it is a no-pivot method.

If a system is singular, nearly singular, strongly nonsymmetric, or otherwise
likely to need row exchanges, use `lu(D)` first and compare against the compact
path before relying on `lu!(copy(D))`.

### Generic Banded Routines

The generic routines are small no-pivot banded LU and triangular-solve
implementations following the standard banded algorithm described by Golub and
Van Loan. They operate on any `AbstractMatrix` using ordinary scalar indexing.

`banded_lu!(A, p, q)` assumes that `A` has at most `p` subdiagonals and `q`
superdiagonals. It first checks that all entries outside those bands are zero,
then overwrites `A` with the LU factors: the strict lower band stores the
multipliers of the unit-lower factor `L`, while the diagonal and upper band
store `U`.

After factorisation, the reference solve is:

```julia
banded_tril_solve!(A, rhs, p)
banded_triu_solve(A, rhs, q)
```

or equivalently:

```julia
ldiv!(A, rhs, p, q)
```

These routines are intentionally simple. They are good for validation,
experiments, and reading the algorithm, but they do not exploit the compact row
layout or generated kernels used by `DiffMatrix`.

### Compact `DiffMatrix` Path

`lu!(D::DiffMatrix)` applies the same no-pivot banded factorisation directly to
the compact coefficient storage. No dense matrix or LAPACK workspace is formed.
The factors replace the original differentiation coefficients, so copy the
operator first if it is still needed.

`ldiv!(D, rhs)` then performs generated forward and backward substitution. The
generated triangular solves specialize on the stencil width stored in the
`DiffMatrix` type. Interior substitution work is emitted as fixed-width
unrolled code, while boundary rows use scalar indexing. If the matrix was built
with `optimise=true`, the diagonal of `U` is stored inverted during
factorisation, so the back-substitution multiplies by the stored reciprocal
instead of dividing.

The compact path is the one measured as **FDGrids** in the
[Benchmarks](benchmarks.md) page. It is the recommended path for repeated
solves when no-pivot LU is appropriate for the system.

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

## Boundary Value Problem

A common workflow is to assemble a differential operator, overwrite boundary
rows with the boundary conditions, and solve the resulting banded system.

For example, solve the Dirichlet problem

```math
u''(x) = -\pi^2 \sin(\pi x), \qquad x \in [-1,1],
```

with

```math
u(-1) = 0, \qquad u(1) = 0.
```

The exact solution is `u(x) = sin(pi*x)`.

```@example linear-solves
M = 64
g = grid(M, -1, 1, GaussLobattoGrid())

L = DiffMatrix(g.xs, 7, 2)

# Replace the first and last equations with Dirichlet boundary conditions.
# The function basis_vector is provided by this package.
L[1,   :] .= basis_vector(1, M)
L[end, :] .= basis_vector(M, M)

rhs = -(π^2) .* sin.(π .* g.xs)
rhs[1]   = 0
rhs[end] = 0

u = ldiv!(lu!(L), copy(rhs))
maximum(abs, u .- sin.(π .* g.xs))
```

This row-overwrite pattern is the intended way to impose strong boundary
conditions with the compact operator. If you still need the original derivative
matrix later, copy it before overwriting rows or construct a fresh operator for
the solve.

!!! note
    Assigning entries does not change the compact banded layout of a
    `DiffMatrix`. Values assigned outside the stored stencil for a row are
    ignored and remain structural zeros. Boundary-condition rows should
    therefore use nonzero entries that lie inside the row's existing band, such
    as `basis_vector(1, M)` for the first row and `basis_vector(M, M)` for the
    last row.

## Choosing a Path

Use `lu!(copy(D))` followed by `ldiv!(Dfac, rhs)` for repeated solves when the
system is well behaved without pivoting. This is the path optimized for
`DiffMatrix` storage.

Use `lu(D)` when you want the pivoted LAPACK reference path or want to compare
against LAPACK's `gbtrf!`/`gbtrs!` implementation.

Use `banded_lu!` and the generic triangular solvers when you want a transparent
ordinary-indexing implementation of the Golub--Van Loan banded algorithm. They
are useful for experiments and validation, not as the preferred production path
for compact `DiffMatrix` systems.

The [Benchmarks](benchmarks.md) page compares these three paths across grid
sizes and stencil widths.

See the [API Reference](../api.md#Linear-Solves) for docstrings and
signatures.
