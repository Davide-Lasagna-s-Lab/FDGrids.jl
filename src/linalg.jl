#
# This file contains three linear-solve paths, ordered from most specialised to
# most external:
#
#   1. Generated `DiffMatrix` kernels for the performance-critical compact path.
#   2. Generic dense-matrix banded LU/solve routines for algorithmic reference.
#   3. LAPACK banded LU wrappers for comparison and reference.
#
# Production code that repeatedly solves with `DiffMatrix` should use the first
# path: `lu!(copy(D))` followed by `ldiv!(factorised_D, rhs)`.

# ================================================================================
# Compact DiffMatrix path
# ================================================================================

"""
    DiffMatrixLU{T, WIDTH, OPTIMISE}

LU factorisation of a `DiffMatrix` produced by `lu!(D)`.

`factors` stores the compact in-place LU factors inside the original
`DiffMatrix` coefficient storage. No dense workspace is allocated.

If `OPTIMISE` is `true`, the diagonal of `U` is stored inverted during
factorisation so that back-substitution multiplies rather than divides.

This type is the result of the performance-critical compact factorisation path.
"""
struct DiffMatrixLU{T, WIDTH, OPTIMISE}
    factors :: DiffMatrix{T, WIDTH, OPTIMISE}
end

@generated function _banded_lu!(A::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE}
    WD = WIDTH >> 1
    op = OPTIMISE ? :(inv(A[i, i])) : :(A[i, i])
    quote
    n = size(A, 1)
    c = A.coeffs
    @inbounds begin
        # head boundary rows: k = 1,...,WD
        for k = 1:$WD
            Akk = A[k, k]
            for i = k+1:min(k+$WIDTH-1, n)
                A[i, k] = A[i, k] / Akk
            end
            for j = k+1:min(k+$WIDTH-1, n)
                Akj = A[k, j]
                for i = k+1:min(k+$WIDTH-1, n)
                    A[i, j] = A[i, j] - A[i, k]*Akj
                end
            end
        end
        # interior rows: direct coeffs access, loops unrolled on WD
        for k = ($WD+1):(n-$WIDTH)
            Akk = c[(k-1)*$WIDTH + $WD + 1]
            # compute multipliers A[k+δ, k] /= A[k,k]
            Base.Cartesian.@nexprs $WD δ -> begin
                c[(k+δ-1)*$WIDTH + $WD - δ + 1] /= Akk
            end
            # update A[k+δ, k+ε] -= A[k+δ, k] * A[k, k+ε]
            Base.Cartesian.@nexprs $WD ε -> begin
                Akε = c[(k-1)*$WIDTH + $WD + ε + 1]
                Base.Cartesian.@nexprs $WD δ -> begin
                    c[(k+δ-1)*$WIDTH + $WD + ε - δ + 1] -= c[(k+δ-1)*$WIDTH + $WD - δ + 1] * Akε
                end
            end
        end
        # tail boundary rows: k = n-WIDTH+1,...,n-1
        for k = max(n-$WIDTH+1, $WD+1):(n-1)
            Akk = A[k, k]
            for i = k+1:min(k+$WIDTH-1, n)
                A[i, k] = A[i, k] / Akk
            end
            for j = k+1:min(k+$WIDTH-1, n)
                Akj = A[k, j]
                for i = k+1:min(k+$WIDTH-1, n)
                    A[i, j] = A[i, j] - A[i, k]*Akj
                end
            end
        end
    end
    if $OPTIMISE
        for i = 1:n
            A[i, i] = inv(A[i, i])
        end
    end
    return A
    end
end

"""
    lu!(A::DiffMatrix) -> DiffMatrixLU

Factor a `DiffMatrix` in place using a no-pivot banded LU and return a
`DiffMatrixLU` wrapping the factorised compact storage.

The compact coefficient array of `A` is overwritten with the LU factors.
When `OPTIMISE=true` (the default), the diagonal entries of `U` are stored
inverted to replace divisions with multiplications during triangular solves.
Copy the matrix first if the original differentiation operator is still needed.

# Examples
```julia
Dfac = lu!(copy(D))
ldiv!(Dfac, rhs)
```
"""
LinearAlgebra.lu!(A::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} =
    DiffMatrixLU(_banded_lu!(A))

"""
    ldiv!(F::DiffMatrixLU, b) -> b

Solve in place using a `DiffMatrixLU` factorisation produced by `lu!`.

This dispatch uses generated triangular solves specialised on the stencil width
encoded in the `DiffMatrix` type. It is the preferred performance-critical solve
path in this file.
"""
LinearAlgebra.ldiv!(F::DiffMatrixLU, b::AbstractVector) =
    _banded_triu_solve!(F.factors, _banded_tril_solve!(F.factors, b))

"""
    full(F::DiffMatrixLU) -> Matrix

Return the dense matrix representation of the compact LU factors stored in `F`.
"""
full(F::DiffMatrixLU) = full(F.factors)

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

# ================================================================================
# Generic banded reference routines
# ================================================================================

"""
    _banded_lu!(A, p, q, optimise=false; check=true) -> A

Factor a dense banded matrix in place without pivoting.

`p` is the number of subdiagonals and `q` is the number of superdiagonals. The
strictly lower part of the band stores the multipliers for `L`, while the upper
part stores `U`. If `optimise=true`, the diagonal of `U` is replaced by its
reciprocal so the specialised triangular solve can multiply instead of divide.

If `check=true` (the default), an O(N²) pass verifies that all entries outside
the declared band are zero before factorising. Set `check=false` to skip this
when the band structure is guaranteed by construction.

It operates on ordinary matrix indexing and is intended as an algorithmic
reference. The compact `DiffMatrix` methods above use the same factorisation
idea but avoid dense storage and generate unrolled triangular solves.
"""
function _banded_lu!(A::AbstractMatrix, p::Int, q::Int, optimise::Bool=false; check::Bool=true)
    n = size(A, 1)
    if check
        for i = 1:n, j = 1:n
            if (i - j) > p || (j - i) > q
                A[i, j] == 0 || throw(ArgumentError("invalid band size"))
            end
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
method above is the preferred performance-critical path.
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
method above is the preferred performance-critical path.
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
    BandedMatrixLU{M<:AbstractMatrix}

Factorisation of a banded matrix produced by `lu!(A, p, q)`.

Stores the in-place LU factors alongside the sub- and super-diagonal counts
`p` and `q` so that `ldiv!` does not need them as separate arguments.

# Fields

- `factors` — the matrix `A` overwritten in-place with its no-pivot LU factors:
               the strict lower band holds the `L` multipliers and the upper
               band (including diagonal) holds `U`.
- `p`       — number of subdiagonals declared at factorisation time.
- `q`       — number of superdiagonals declared at factorisation time.
"""
struct BandedMatrixLU{M<:AbstractMatrix}
    factors :: M
    p       :: Int
    q       :: Int
end

"""
    lu!(A::AbstractMatrix, p, q; optimise=false, check=true) -> BandedMatrixLU

Factorise a banded matrix in place and return a [`BandedMatrixLU`](@ref).

`p` is the number of subdiagonals and `q` is the number of superdiagonals.
`A` is overwritten with the no-pivot LU factors. The returned `BandedMatrixLU`
wraps `A` together with `p` and `q` so that `ldiv!` can be called without
repeating the bandwidth arguments.

If `check=true` (the default), an O(N²) pass verifies that all entries outside
the declared band are zero. Pass `check=false` when the band structure is
guaranteed by construction to skip this cost.

This routine is intended for reference and experimentation with ordinary
matrices. Performance-critical code using `DiffMatrix` should prefer
`lu!(copy(D))` and `ldiv!(Dfac, rhs)`.

# Examples

```julia
A   = Float64[2 -1 0; -1 2 -1; 0 -1 2]
F   = lu!(A, 1, 1)
rhs = [1.0, 0.0, 1.0]
ldiv!(F, rhs)
```
"""
LinearAlgebra.lu!(A::AbstractMatrix, p::Int, q::Int; optimise::Bool=false, check::Bool=true) =
    BandedMatrixLU(_banded_lu!(A, p, q, optimise; check), p, q)

"""
    ldiv!(F::BandedMatrixLU, b) -> b

Solve the banded system represented by a `BandedMatrixLU` factorisation in
place, overwriting `b` with the solution.

The factorisation must have been produced by `lu!(A, p, q)` for a compatible
matrix. Bandwidth parameters are read from `F` directly.
"""
LinearAlgebra.ldiv!(F::BandedMatrixLU, b::AbstractVector) =
    _banded_triu_solve!(F.factors, _banded_tril_solve!(F.factors, b, F.p), F.q)

"""
    ldiv!(A::AbstractMatrix, b, p, q) -> b

Solve a banded system after `A` has already been factorised by `lu!`.

`p` and `q` must match the bandwidths used during factorisation. The solution is
written in place into `b`.

This dispatch is kept for reference routines that store factors in an ordinary
matrix. Performance-critical `DiffMatrix` solves should use the compact methods
above.
"""
LinearAlgebra.ldiv!(A::AbstractMatrix, b::AbstractVector, p::Int, q::Int) =
    _banded_triu_solve!(A, _banded_tril_solve!(A, b, p), q)

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
path for the compact generated implementation above.
"""
function LinearAlgebra.ldiv!(lu::DiffMatrixLULapack{T, WIDTH}, x::AbstractVector{T}) where {T, WIDTH}
    WD = WIDTH-1
    return LinearAlgebra.LAPACK.gbtrs!('N', WD, WD, size(lu.factors, 2), lu.factors, lu.ipiv, x)
end
