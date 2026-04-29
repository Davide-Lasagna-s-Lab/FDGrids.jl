"""
    DiffMatrix{T, WIDTH, OPTIMISE, A<:AbstractMatrix{T}} <: AbstractMatrix{T}

Compact representation of a banded finite-difference differentiation matrix
with stencil width `WIDTH`.

Coefficients are stored column-major in `coeffs` (`WIDTH × N`): column `i`
contains the `WIDTH` stencil weights for output row `i` of D.

`OPTIMISE` controls whether the diagonal of the LU factor U is stored inverted
(saves one division per element in the triangular solve).

The storage type `A` is `Matrix{T}` on the CPU and `CuMatrix{T}` on the GPU.
Use `CUDA.cu(d)` to move a CPU `DiffMatrix` to the GPU.
"""
struct DiffMatrix{T, WIDTH, OPTIMISE, A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    coeffs :: A   # finite difference weights — Matrix{T} on CPU, CuMatrix{T} on GPU

    """
        DiffMatrix(xs, width, order; optimise=true, eltype=Float64)

    Construct a finite-difference differentiation matrix on `xs` of the given
    stencil `width` and derivative `order`. Always constructs on the CPU —
    use `CUDA.cu(d)` to move to the GPU (requires CUDA.jl).

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

        coeffs = eltype.(get_coeffs(xs, width, order))

        return new{eltype, width, optimise, typeof(coeffs)}(coeffs)
    end

    # Internal constructor: wraps an existing coeffs array directly.
    # Used by copy() and similar() — not part of the public API.
    DiffMatrix{T, WIDTH, OPTIMISE, A}(coeffs::A) where {T, WIDTH, OPTIMISE, A<:AbstractMatrix{T}} =
        new{T, WIDTH, OPTIMISE, A}(coeffs)
end


# ================================================================================
# AbstractMatrix interface
# ================================================================================

Base.size(d::DiffMatrix) = (size(d.coeffs, 2), size(d.coeffs, 2))
Base.IndexStyle(::DiffMatrix) = IndexCartesian()

# NOTE: when coeffs is a CuMatrix, each getindex/setindex! call does a scalar
# device-to-host memory transfer — correct but very slow. These methods are
# CPU-only operations (indexing, full(), linalg.jl) and should never be called
# in a hot loop on a GPU DiffMatrix.
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

function Base.similar(d::DiffMatrix{T, WIDTH, OPTIMISE, A}, ::Type{S}=T) where {T, S, WIDTH, OPTIMISE, A}
    c = similar(d.coeffs, S)
    return DiffMatrix{S, WIDTH, OPTIMISE, typeof(c)}(c)
end

Base.copy(d::DiffMatrix{T, WIDTH, OPTIMISE, A}) where {T, WIDTH, OPTIMISE, A} =
    DiffMatrix{T, WIDTH, OPTIMISE, A}(copy(d.coeffs))

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
# Broadcasting — CPU only. GPU differentiation goes through mul! dispatch.
# ================================================================================

struct DiffMatrixStyle{T, WIDTH, OPTIMISE} <: Broadcast.BroadcastStyle end

# Only register the CPU (Matrix-backed) variant for broadcasting.
Base.BroadcastStyle(::Type{<:DiffMatrix{T, WIDTH, OPTIMISE, <:Matrix}}) where {T, WIDTH, OPTIMISE} =
    DiffMatrixStyle{T, WIDTH, OPTIMISE}()

# broadcast with scalar: α * D
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, s::DiffMatrixStyle) = s

# broadcast with a vector: e.g. D .* v
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{1}, s::DiffMatrixStyle) = s

# broadcast with diagonal matrix: D + Diagonal(v)
Base.BroadcastStyle(::LinearAlgebra.StructuredMatrixStyle{<:LinearAlgebra.Diagonal}, s::DiffMatrixStyle) = s

# broadcast between two DiffMatrices (e.g. D1 + D2)
Base.BroadcastStyle(::DiffMatrixStyle{T1, W1, O1}, ::DiffMatrixStyle{T2, W2, O2}) where {T1, T2, W1, W2, O1, O2} =
    DiffMatrixStyle{promote_type(T1, T2), max(W1, W2), O1 || O2}()

# allocate new output
Base.similar(bc::Base.Broadcast.Broadcasted{DiffMatrixStyle{T, WIDTH, OPTIMISE}}, ::Type{S}) where {T, WIDTH, OPTIMISE, S} =
    DiffMatrix{S, WIDTH, OPTIMISE, Matrix{S}}(Matrix{S}(undef, WIDTH, axes(bc)[1][end]))
