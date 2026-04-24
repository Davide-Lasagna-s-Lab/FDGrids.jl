import LinearAlgebra

"""
    DiffMatrix{T, WIDTH, OPTIMISE} <: AbstractMatrix{T}

Compact representation of a banded finite-difference differentiation matrix
with stencil width `WIDTH`.

Coefficients are stored column-major in `coeffs` (`WIDTH × N`): column `i`
contains the `WIDTH` stencil weights for output row `i` of D.

`OPTIMISE` controls whether the diagonal of the LU factor U is stored inverted
(saves one division per element in the triangular solve).
"""
struct DiffMatrix{T, WIDTH, OPTIMISE} <: AbstractMatrix{T}
    coeffs::Matrix{T}   # stencil weights, WIDTH×N, column i = stencil for row i
      buff::Vector{T}   # small buffer for the matvec code

    function DiffMatrix(xs      ::AbstractVector,
                        width   ::Int,
                        order   ::Int,
                        optimise::Bool    = true,
                                ::Type{T} = Float64) where {T}
        3 ≤ width          || throw(ArgumentError("width must be ≥ 3"))
        width % 2 == 1     || throw(ArgumentError("width must be odd"))
        width ≤ length(xs) || throw(ArgumentError("width must not exceed the number of grid points"))

        # Coefficients are organised in column-major format: column i contains
        # the stencil weights for output row i of D.
        coeffs = get_coeffs(xs, width, order)

        return new{T, width, optimise}(T.(coeffs), zeros(T, width))
    end
end


# ================================================================================
# AbstractMatrix interface
# ================================================================================

Base.size(d::DiffMatrix) = (size(d.coeffs, 2), size(d.coeffs, 2))
Base.IndexStyle(::DiffMatrix) = IndexCartesian()

function Base.getindex(d::DiffMatrix{T, WIDTH}, i::Int, j::Int) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    offset = i ≤              HWIDTH ?          HWIDTH - i + 1 :
             i > size(d, 1) - HWIDTH ? size(d, 1) - HWIDTH - i : 0
    m, n   = HWIDTH + j - i + 1 - offset, i
    return checkbounds(Bool, d.coeffs, m, n) ? d.coeffs[m, n] : zero(T)
end

function Base.setindex!(d::DiffMatrix{T, WIDTH}, v, i::Int, j::Int) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    offset = i ≤              HWIDTH ?          HWIDTH - i + 1 :
             i > size(d, 1) - HWIDTH ? size(d, 1) - HWIDTH - i : 0
    m, n   = HWIDTH + j - i + 1 - offset, i
    return checkbounds(Bool, d.coeffs, m, n) ? (d.coeffs[m, n] = T(v)) : T(v)
end

function Base.similar(d    ::DiffMatrix{T, WIDTH, OPTIMISE},
                      ::Type{S}                     = T,
                      _size::Tuple{Vararg{Int64, 2}} = size(d)) where {T, S, WIDTH, OPTIMISE}
    return DiffMatrix(zeros(Float64, _size[1]), WIDTH, 1, OPTIMISE, S)
end

function Base.copy(d::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    d_ = similar(d)
    d_.coeffs .= d.coeffs
    return d_
end

"""
    full(A::DiffMatrix) -> Matrix{T}

Expand the compact stencil representation into a dense `N×N` matrix.
"""
function full(A::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    N   = size(A.coeffs, 2)
    out = zeros(T, N, N)
    @simd for i in 1:N
        left = clamp(i - (WIDTH >> 1), 1, N - WIDTH + 1)
        for p in 1:WIDTH
            out[i, left + p - 1] = A.coeffs[p, i]
        end
    end
    return out
end


# ================================================================================
# Broadcasting
# ================================================================================

struct DiffMatrixStyle{T, WIDTH, OPTIMISE} <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:DiffMatrix{T, WIDTH, OPTIMISE}}) where {T, WIDTH, OPTIMISE} =
    DiffMatrixStyle{T, WIDTH, OPTIMISE}()

Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0},
                     s::DiffMatrixStyle{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} = s

Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{1},
                     s::DiffMatrixStyle{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} = s

Base.BroadcastStyle(::LinearAlgebra.StructuredMatrixStyle{<:LinearAlgebra.Diagonal},
                     s::DiffMatrixStyle{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} = s

Base.BroadcastStyle(::DiffMatrixStyle{T1, W1, OPTIMISE},
                    ::DiffMatrixStyle{T2, W2, OPTIMISE}) where {T1, T2, W1, W2, OPTIMISE} =
    DiffMatrixStyle{promote_type(T1, T2), max(W1, W2), OPTIMISE}()

function Base.similar(bc  ::Base.Broadcast.Broadcasted{<:DiffMatrixStyle{T, WIDTH, OPTIMISE}},
                      ::Type{S}) where {T, WIDTH, OPTIMISE, S}
    s = axes(bc)[1][end]
    DiffMatrix(zeros(Float64, s), WIDTH, 1, OPTIMISE, S)
end
