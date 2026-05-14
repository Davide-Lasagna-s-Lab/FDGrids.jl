#
# This file contains three linear-solve paths, ordered from most external to
# most specialised:
#
#   1. LAPACK banded LU wrappers for comparison and reference.
#   2. Generic dense-matrix banded LU/solve routines for algorithmic reference.
#   3. Generated `DiffMatrix` kernels for the performance-critical compact path.
#
# Production code that repeatedly solves with `DiffMatrix` should use the last
# path: `lu!(copy(D))` followed by `ldiv!(factorised_D, rhs)`.

# ================================================================================
# LAPACK reference path
# ================================================================================

"""
    to_banded_format(D::DiffMatrix) -> Matrix

Convert `D` to the banded storage layout expected by LAPACK's `gbtrf!`.

The returned array has `3*WIDTH-2` rows and `size(D,2)` columns. It deliberately
uses the LAPACK work layout rather than the minimal mathematical bandwidth, so it
can be passed directly to `LinearAlgebra.LAPACK.gbtrf!`.

This conversion is part of the LAPACK reference path. It allocates and expands
the compact coefficients into LAPACK's banded workspace, so it is useful for
validation and comparison rather than the fastest `DiffMatrix` solve path.
"""
function to_banded_format(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    # number of rows
    n = size(D, 1)

    # allocate output
    out = zeros(T, 3*WIDTH-2, n)

    for _i = 1:(2*WIDTH-1)
        for _j = 1:n
            # indices in global coordinates
            i, j = _j + (_i - WIDTH), _j
            val = checkbounds(Bool, D, i, j) ? D[i, j] : zero(T)
            out[_i + WIDTH-1, _j] = val
        end
    end

    return out
end

"""
    DiffMatrixLULapack{T, WIDTH, F}

LU factorisation of a `DiffMatrix` produced by `lu(D)`.

`factors` stores LAPACK's banded LU array and `ipiv` stores the pivot vector.
This type belongs to the LAPACK reference path.
"""
struct DiffMatrixLULapack{T, WIDTH, F}
    factors::F
    ipiv::Vector{Int}
end

"""
    lu(D::DiffMatrix) -> DiffMatrixLULapack

Compute a pivoted banded LU factorisation of `D` using LAPACK.

This is the allocating, LAPACK-backed factorisation path. It preserves the
compact `DiffMatrix` input and stores the factors in LAPACK banded format.

Use this when comparing against LAPACK or when pivoting is desired. For
performance-critical repeated solves with `DiffMatrix`, prefer the compact
in-place path `lu!(copy(D))` followed by `ldiv!(factorised_D, rhs)`.
"""
function LinearAlgebra.lu(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    WD = WIDTH-1
    factors, ipiv = LinearAlgebra.LAPACK.gbtrf!(WD, WD, size(D, 2), to_banded_format(D))
    return DiffMatrixLULapack{T, WIDTH, typeof(factors)}(factors, ipiv)
end

"""
    ldiv!(lu::DiffMatrixLULapack, x) -> x

Solve the linear system represented by a `DiffMatrixLULapack` factorisation in place.

`x` is overwritten with the solution. The factorisation must have been produced
by `lu(D)` for a compatible `DiffMatrix`.

This solves through LAPACK's banded storage and is mainly a reference/comparison
path for the compact generated implementation below.
"""
function LinearAlgebra.ldiv!(lu::DiffMatrixLULapack{T, WIDTH}, x::AbstractVector{T}) where {T, WIDTH}
    WD = WIDTH-1
    return LinearAlgebra.LAPACK.gbtrs!('N', WD, WD, size(lu.factors, 2), lu.factors, lu.ipiv, x)
end

# ================================================================================
# Generic banded reference routines
# ================================================================================

"""
    _banded_lu!(A, p, q, optimise=false) -> A

Factor a dense banded matrix in place without pivoting.

`p` is the number of subdiagonals and `q` is the number of superdiagonals. The
strictly lower part of the band stores the multipliers for `L`, while the upper
part stores `U`. If `optimise=true`, the diagonal of `U` is replaced by its
reciprocal so the specialised triangular solve can multiply instead of divide.

This routine is a small reference implementation of the banded LU algorithm
from Golub and Van Loan. It checks that entries outside the declared band are
zero before factorising.

It operates on ordinary matrix indexing and is intended as an algorithmic
reference. The compact `DiffMatrix` methods below use the same factorisation
idea but avoid dense storage and generate unrolled triangular solves.
"""
function _banded_lu!(A::AbstractMatrix, p::Int, q::Int, optimise::Bool=false)
    n = size(A, 1)
    # Sanity check: no elements outside the bands is different from zero
    for i = 1:n, j=1:n
        # if out of bands
        if ((i-j) > p || (j-i) > q)
            A[i, j] == 0 || throw(ArgumentError("invalid band size"))
        end
    end
    @inbounds for k = 1:n-1
        for i = k+1:min(k+p, n)
            A[i, k] = A[i, k]/A[k, k]
        end
        for j = k+1:min(k+q, n)
            for i = k+1:min(k+p, n)
                A[i, j] = A[i, j] - A[i, k]*A[k, j]
            end
        end
    end
    # Pre calculate the inverse of the diagonal of the factor U.
    # This avoid doing one division per element.
    if optimise
        for i = 1:n
            A[i, i] = inv(A[i, i])
        end
    end
    return A
end

"""
    _banded_tril_solve!(L, b, p) -> b

Forward-substitute through the unit-lower triangular factor stored in `L`.

Only the first `p` subdiagonals are inspected. The right-hand side `b` is
overwritten with the intermediate solution.

This is the generic reference implementation. The specialised `DiffMatrix`
method below is the preferred performance-critical path.
"""
function _banded_tril_solve!(L::AbstractMatrix, b::AbstractVector, p::Int)
    n = size(L, 1)
    @inbounds for j = 1:n
        for i = j+1:min(j+p, n)
            b[i] = b[i] - L[i, j]*b[j]
        end
    end
    return b
end

"""
    _banded_triu_solve!(U, b, q) -> b

Back-substitute through the upper triangular factor stored in `U`.

Only the first `q` superdiagonals are inspected. The right-hand side `b` is
overwritten with the final solution.

This is the generic reference implementation. The specialised `DiffMatrix`
method below is the preferred performance-critical path.
"""
function _banded_triu_solve!(U::AbstractMatrix, b::AbstractVector, q::Int)
    n = size(U, 1)
    @inbounds for j = n:-1:1
        b[j] = b[j]/U[j, j]
        for i = max(1, j-q):j-1
            b[i] = b[i] - U[i, j]*b[j]
        end
    end
    return b
end

"""
    ldiv!(A::AbstractMatrix, b, p, q) -> b

Solve a banded system after `A` has already been factorised by `banded_lu!`.

`p` and `q` must match the bandwidths used during factorisation. The solution is
written in place into `b`.

This dispatch is kept for reference routines that store factors in an ordinary
matrix. Performance-critical `DiffMatrix` solves should use the compact methods
below.
"""
LinearAlgebra.ldiv!(A::AbstractMatrix, b::AbstractVector, p::Int, q::Int) = 
    _banded_triu_solve!(A, _banded_tril_solve!(A, b, p), q)

"""
    banded_lu!(A, p, q; optimise=false) -> A

Public wrapper around the generic reference banded LU factorisation.

`p` is the number of subdiagonals and `q` is the number of superdiagonals.
The matrix `A` is overwritten with the no-pivot LU factors produced by the
reference implementation `_banded_lu!`.

This routine is intended for reference and experimentation with ordinary
matrices. Performance-critical code using `DiffMatrix` should prefer
`lu!(copy(D))` and `ldiv!(Dfac, rhs)`.
"""
banded_lu!(A::AbstractMatrix, p::Int, q::Int; optimise::Bool=false) =
    _banded_lu!(A, p, q, optimise)

"""
    banded_tril_solve!(L, b, p) -> b

Public wrapper around the generic reference forward-substitution routine.

`L` is assumed to contain the unit-lower triangular factor produced by
`banded_lu!`, and only the first `p` subdiagonals are inspected. The vector `b`
is overwritten.
"""
banded_tril_solve!(L::AbstractMatrix, b::AbstractVector, p::Int) =
    _banded_tril_solve!(L, b, p)

"""
    banded_triu_solve(U, b, q) -> b

Public wrapper around the generic reference back-substitution routine.

`U` is assumed to contain the upper triangular factor produced by `banded_lu!`,
and only the first `q` superdiagonals are inspected. The vector `b` is
overwritten.

This exported name is kept for compatibility with the existing public API even
though the function mutates `b`.
"""
banded_triu_solve(U::AbstractMatrix, b::AbstractVector, q::Int) =
    _banded_triu_solve!(U, b, q)


# ================================================================================
# Compact DiffMatrix implementation
# ================================================================================

"""
    lu!(A::DiffMatrix) -> A

Factor a `DiffMatrix` in place with the no-pivoting banded LU routine.

The factorisation overwrites the compact coefficients of `A`. If the
`OPTIMISE` type parameter is true, the diagonal of `U` is inverted during
factorisation to save divisions during repeated triangular solves.

This is the preferred factorisation path for performance-critical code using
`DiffMatrix`. It avoids LAPACK workspace conversion and keeps the factors in the
same compact coefficient storage. Copy the matrix first if the original
differentiation operator is still needed.

# Examples
```julia
Dfac = lu!(copy(D))
ldiv!(Dfac, rhs)
```
"""
LinearAlgebra.lu!(A::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} =
    _banded_lu!(A, WIDTH-1, WIDTH-1, OPTIMISE)

"""
    ldiv!(A::DiffMatrix, b) -> b

Solve in place using a `DiffMatrix` that has already been factorised by `lu!`.

This dispatch uses generated triangular solves specialised on the stencil width
encoded in the `DiffMatrix` type. It is the preferred performance-critical solve
path in this file.
"""
LinearAlgebra.ldiv!(A::DiffMatrix{T, WIDTH}, b::AbstractVector) where {T, WIDTH} = 
    _banded_triu_solve!(A, _banded_tril_solve!(A, b))

"""
    _banded_tril_solve!(L::DiffMatrix, b) -> b

Generated forward-substitution kernel for a compact `DiffMatrix` LU factor.

The body loop is unrolled using the stencil width encoded in the matrix type,
while the boundary rows fall back to scalar indexing.
"""
@generated function _banded_tril_solve!(L::DiffMatrix{T, WIDTH}, b::AbstractVector) where {T, WIDTH}
    WD = WIDTH>>1
    quote
    n = size(L, 1)
    @inbounds begin
        for j = 1:($WD-1)
            for i = j+1:min(j+$WIDTH - 1, n)
                b[i] = b[i] - L[i, j]*b[j]
            end
        end
        for j = $WD:(n-$WIDTH)
            Base.Cartesian.@nexprs $WD i_ -> begin
                b[j+i_] = muladd(L.coeffs[(i_+j-1)*$WIDTH + ($WD + 1 - i_)], -b[j], b[j+i_])
            end
        end
        for j = max(n-$WIDTH+1, $WD):n
            for i = j+1:min(j+$WIDTH - 1, n)
                b[i] = b[i] - L[i, j]*b[j]
            end
        end
    end
    return b
    end
end

"""
    _banded_triu_solve!(U::DiffMatrix, b) -> b

Generated back-substitution kernel for a compact `DiffMatrix` LU factor.

If `U` has `OPTIMISE=true`, the diagonal entries are assumed to store
reciprocals and the solve multiplies by them. Otherwise the solve divides by
the stored diagonal entries.
"""
@generated function _banded_triu_solve!(U::DiffMatrix{T, WIDTH, OPTIMISE}, b::AbstractVector) where {T, WIDTH, OPTIMISE}
    WD = WIDTH>>1
    op = OPTIMISE == true ? Base.:* : Base.:/
    quote
    n = size(U, 1)
    @inbounds begin
        for j = n:-1:(n-$WD+1)
            b[j] = $op(b[j], U[j, j])
            for i = max(1, j-$WIDTH+1):j-1
                b[i] = b[i] - U[i, j]*b[j]
            end
        end
        for j = (n-$WD):-1:($WIDTH+1)
            b[j] = $op(b[j], U.coeffs[(j-1)*$WIDTH + ($WD+1)])
            Base.Cartesian.@nexprs $WD i_ -> begin
                b[j-i_] = muladd(U.coeffs[(j-i_-1)*$WIDTH + ($WD+1+i_)], -b[j], b[j-i_])
            end
        end
        for j = min($WIDTH, n-$WD):-1:1
            b[j] = $op(b[j], U[j, j])
            for i = max(1, j-$WIDTH+1):j-1
                b[i] = b[i] - U[i, j]*b[j]
            end
        end
    end
    return b
    end
end
