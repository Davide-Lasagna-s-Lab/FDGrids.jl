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
        3 ≤ width          || throw(ArgumentError("width must be greater than 3"))
        width % 2 == 1     || throw(ArgumentError("width must be odd"))
        width ≤ length(xs) || throw(ArgumentError("width must not be greater than number of grid points"))

        coeffs = eltype.(get_coeffs(xs, width, order))

        return new{eltype, width, optimise, typeof(coeffs)}(coeffs)
    end

    # Internal constructor: wraps an existing coeffs array directly.
    # Used by copy() and similar() — not part of the public API.
    DiffMatrix{T, WIDTH, OPTIMISE, A}(coeffs::A) where {T, WIDTH, OPTIMISE, A<:AbstractMatrix{T}} =
        new{T, WIDTH, OPTIMISE, A}(coeffs)
end

Base.size(d::DiffMatrix) = (size(d.coeffs, 2), size(d.coeffs, 2))
Base.IndexStyle(d::DiffMatrix) = IndexCartesian()

# NOTE: when coeffs is a CuMatrix, each getindex/setindex! call does a scalar
# device-to-host memory transfer — correct but very slow. These methods are
# CPU-only operations (indexing, full(), linalg.jl) and should never be called
# in a hot loop on a GPU DiffMatrix.
function Base.getindex(d::DiffMatrix{T, WIDTH}, i::Int, j::Int) where {T, WIDTH}
    offset = i ≤              WIDTH >> 1 ?          WIDTH>>1 - i + 1 :
             i > size(d, 1) - WIDTH >> 1 ? size(d, 1) - WIDTH>>1 - i : 0
    m, n = WIDTH>>1 + j - i + 1 - offset, i
    return checkbounds(Bool, d.coeffs, m, n) ? d.coeffs[m, n] : zero(T)
end

function Base.setindex!(d::DiffMatrix{T, WIDTH}, v, i::Int, j::Int) where {T, WIDTH}
    offset = i ≤              WIDTH >> 1 ?          WIDTH>>1 - i + 1 :
             i > size(d, 1) - WIDTH >> 1 ? size(d, 1) - WIDTH>>1 - i : 0
    m, n = WIDTH>>1 + j - i + 1 - offset, i
    return checkbounds(Bool, d.coeffs, m, n) ? (d.coeffs[m, n] = T(v)) : T(v)
end

Base.similar(d::DiffMatrix{T, WIDTH, OPTIMISE, A}, ::Type{S}=T) where {T, S, WIDTH, OPTIMISE, A} =
    DiffMatrix{S, WIDTH, OPTIMISE, A}(similar(d.coeffs, S))

Base.copy(d::DiffMatrix{T, WIDTH, OPTIMISE, A}) where {T, WIDTH, OPTIMISE, A} =
    DiffMatrix{T, WIDTH, OPTIMISE, A}(copy(d.coeffs))

function full(A::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    N = size(A, 1)
    out = zeros(T, N, N)
    for i = 1:N, j = 1:N
        out[i, j] = A[i, j]
    end
    return out
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Broadcasting — CPU only. GPU differentiation goes through mul! dispatch.
struct DiffMatrixStyle{T, WIDTH, OPTIMISE} <: Broadcast.BroadcastStyle end

# main broadcasting entry point
Base.BroadcastStyle(::Type{<:DiffMatrix{T, WIDTH, OPTIMISE, <:Matrix}}) where {T, WIDTH, OPTIMISE} =
    DiffMatrixStyle{T, WIDTH, OPTIMISE}()

# broadcast with scalar: α * D
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, s::DiffMatrixStyle) = s

# broadcast with another matrix: D1 + D2 (but no D1 * D2)
Base.BroadcastStyle(::LinearAlgebra.StructuredMatrixStyle{<:LinearAlgebra.Diagonal}, s::DiffMatrixStyle) = s

# broadcast with diffent parameters
Base.BroadcastStyle(::DiffMatrixStyle{T1, W1, O1}, ::DiffMatrixStyle{T2, W2, O2}) where {T1, T2, W1, W2, O1, O2} =
    DiffMatrixStyle{promote_type(T1, T2), max(W1, W2), O1 || O2}()

# allocate new output
Base.similar(bc::Base.Broadcast.Broadcasted{DiffMatrixStyle{T, WIDTH, OPTIMISE}}, ::Type{S}) where {T, WIDTH, OPTIMISE, S} =
    DiffMatrix{S, WIDTH, OPTIMISE, Matrix{S}}(Matrix{S}(undef, WIDTH, axes(bc)[1][end]))