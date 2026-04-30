"""
    DiffMatrix{T, WIDTH, OPTIMISE} <: AbstractMatrix{T}

Compact representation of a banded finite-difference differentiation matrix
with stencil width `WIDTH`.

Coefficients are stored in a flat vector `coeffs` of length `N*WIDTH`: for output
row `i`, the WIDTH stencil weights occupy `coeffs[(i-1)*WIDTH+1 : i*WIDTH]`.

`OPTIMISE` controls whether the diagonal of the LU factor U is stored inverted
(saves one division per element in the triangular solve).
"""
struct DiffMatrix{T, WIDTH, OPTIMISE} <: AbstractMatrix{T}
    coeffs :: Vector{T}

    """
        DiffMatrix(xs, width, order; optimise=true, eltype=Float64)

    Construct a finite-difference differentiation matrix on `xs` of the given
    stencil `width` and derivative `order`.

    # Keyword Arguments
    - `optimise::Bool=true`: pre-invert the diagonal of U during LU factorisation.
    - `eltype::Type=Float64`: element type of the coefficients.
    """
    function DiffMatrix(xs::AbstractVector, width::Int, order::Int;
                        optimise::Bool = true,
                        eltype::Type   = Float64)
        3 ≤ width          || throw(ArgumentError("width must be ≥ 3"))
        width % 2 == 1     || throw(ArgumentError("width must be odd"))
        width ≤ length(xs) || throw(ArgumentError("width must not exceed the number of grid points"))

        coeffs = eltype.(vec(get_coeffs(xs, width, order)))

        return new{eltype, width, optimise}(coeffs)
    end

    DiffMatrix{T, WIDTH, OPTIMISE}(coeffs::Vector{T}) where {T, WIDTH, OPTIMISE} =
        new{T, WIDTH, OPTIMISE}(coeffs)
end


# ================================================================================
# AbstractMatrix interface
# ================================================================================

function Base.size(d::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    N = length(d.coeffs) ÷ WIDTH
    return (N, N)
end

Base.IndexStyle(::DiffMatrix) = IndexCartesian()

function Base.getindex(d::DiffMatrix{T, WIDTH}, i::Int, j::Int) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    offset = i ≤              HWIDTH ?          HWIDTH - i + 1 :
             i > size(d, 1) - HWIDTH ? size(d, 1) - HWIDTH - i : 0
    m = HWIDTH + j - i + 1 - offset
    return (1 ≤ m ≤ WIDTH) ? d.coeffs[(i - 1)*WIDTH + m] : zero(T)
end

function Base.setindex!(d::DiffMatrix{T, WIDTH}, v, i::Int, j::Int) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    offset = i ≤              HWIDTH ?          HWIDTH - i + 1 :
             i > size(d, 1) - HWIDTH ? size(d, 1) - HWIDTH - i : 0
    m = HWIDTH + j - i + 1 - offset
    if 1 ≤ m ≤ WIDTH
        d.coeffs[(i - 1)*WIDTH + m] = T(v)
    end
    return T(v)
end

function Base.similar(d::DiffMatrix{T, WIDTH, OPTIMISE}, ::Type{S}=T) where {T, S, WIDTH, OPTIMISE}
    N = length(d.coeffs) ÷ WIDTH
    return DiffMatrix{S, WIDTH, OPTIMISE}(Vector{S}(undef, N * WIDTH))
end

Base.copy(d::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} =
    DiffMatrix{T, WIDTH, OPTIMISE}(copy(d.coeffs))

"""
    full(A::DiffMatrix) -> Matrix{T}

Expand the compact stencil representation into a dense `N×N` matrix.
"""
function full(A::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    N   = length(A.coeffs) ÷ WIDTH
    out = zeros(T, N, N)
    @simd for i in 1:N
        left = clamp(i - (WIDTH >> 1), 1, N - WIDTH + 1)
        for p in 1:WIDTH
            out[i, left + p - 1] = A.coeffs[(i - 1)*WIDTH + p]
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

Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, s::DiffMatrixStyle) = s
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{1}, s::DiffMatrixStyle) = s
Base.BroadcastStyle(::LinearAlgebra.StructuredMatrixStyle{<:LinearAlgebra.Diagonal}, s::DiffMatrixStyle) = s
Base.BroadcastStyle(::DiffMatrixStyle{T1, W1, O1}, ::DiffMatrixStyle{T2, W2, O2}) where {T1, T2, W1, W2, O1, O2} =
    DiffMatrixStyle{promote_type(T1, T2), max(W1, W2), O1 || O2}()

Base.similar(bc::Base.Broadcast.Broadcasted{DiffMatrixStyle{T, WIDTH, OPTIMISE}}, ::Type{S}) where {T, WIDTH, OPTIMISE, S} =
    DiffMatrix{S, WIDTH, OPTIMISE}(Vector{S}(undef, axes(bc)[1][end] * WIDTH))
