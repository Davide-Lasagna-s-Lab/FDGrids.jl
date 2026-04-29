# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Transform to banded format suitable for the factorisation. This has more
# rows than it should, it is used for reference with the blas implementation
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

struct DiffMatrixLU{T, WIDTH, F}
    factors::F
    ipiv::Vector{Int}
end

function LinearAlgebra.lu(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    WD = WIDTH-1
    factors, ipiv = LinearAlgebra.LAPACK.gbtrf!(WD, WD, size(D, 2), to_banded_format(D))
    return DiffMatrixLU{T, WIDTH, typeof(factors)}(factors, ipiv)
end

function LinearAlgebra.ldiv!(lu::DiffMatrixLU{T, WIDTH}, x::AbstractVector{T}) where {T, WIDTH}
    WD = WIDTH-1
    return LinearAlgebra.LAPACK.gbtrs!('N', WD, WD, size(lu.factors, 2), lu.factors, lu.ipiv, x)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Factorisations and solution of generic banded linear systems without
# pivoting from Golub and Van Loan. These are the reference routines.

# LU factorisation without pivoting of banded system with p bands below
# the diagonal and q above it.
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

function _banded_tril_solve!(L::AbstractMatrix, b::AbstractVector, p::Int)
    n = size(L, 1)
    @inbounds for j = 1:n
        for i = j+1:min(j+p, n)
            b[i] = b[i] - L[i, j]*b[j]
        end
    end
    return b
end

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

# Solve a factorised banded system with p band below the diagonal and 
# q above it. It is assumed A has been factorised using _banded_lu!
LinearAlgebra.ldiv!(A::AbstractMatrix, b::AbstractVector, p::Int, q::Int) = 
    _banded_triu_solve!(A, _banded_tril_solve!(A, b, p), q)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Specialised versions of Golub and Van Loans' methods for DiffMatrices. If the argument
# optimise is true, the diagonal of the factor U is inverted to save one division per 
# loop iteration. For DiffMatrices with small width, this can lead to substantial savings.
# This function operates in place, overwriting A.
LinearAlgebra.lu!(A::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} =
    _banded_lu!(A, WIDTH-1, WIDTH-1, OPTIMISE)

LinearAlgebra.ldiv!(A::DiffMatrix{T, WIDTH}, b::AbstractVector) where {T, WIDTH} = 
    _banded_triu_solve!(A, _banded_tril_solve!(A, b))

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
                b[j+i_] = muladd(L.coeffs[$WD + 1 - i_, i_+j], -b[j], b[j+i_])
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

# Assumes the diagonal of the upper-diagonal matrix is pr`e
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
            b[j] = $op(b[j], U.coeffs[$WD+1, j])
            Base.Cartesian.@nexprs $WD i_ -> begin
                b[j-i_] = muladd(U.coeffs[$WD + 1 + i_, j - i_], -b[j], b[j-i_])
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