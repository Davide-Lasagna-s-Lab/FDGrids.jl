"""
    DiffMatrix{T, WIDTH, OPTIMISE} <: AbstractMatrix{T}

Compact representation of a banded NxN  finite-difference differentiation matrix
with stencil width `WIDTH`.

Coefficients are stored in a flat vector `coeffs` of length `N*WIDTH`: for output
row `i`, the WIDTH stencil weights occupy `coeffs[(i-1)*WIDTH+1 : i*WIDTH]`.

`OPTIMISE` controls whether the diagonal of the LU factor U is stored inverted
(saves one division per element in the triangular solve).

# Examples
```julia
using LinearAlgebra

xs = range(-1, 1; length = 64)
D  = DiffMatrix(xs, 5, 1)   # first derivative, 5-point stencil

u  = sin.(xs)
du = similar(u)
mul!(du, D, u)
```
"""
struct DiffMatrix{T, WIDTH, OPTIMISE} <: AbstractMatrix{T}
    coeffs :: Vector{T}

    """
        DiffMatrix(xs, width, order; optimise=true, eltype=Float64)

    Construct a finite-difference differentiation matrix on `xs` of the given
    stencil `width` and derivative `order`.

    The grid points in `xs` may be non-uniform, but they should be distinct and
    ordered consistently with the arrays to which the operator will be applied.
    Boundary rows use one-sided stencils of the same width.

    # Keyword Arguments
    - `optimise::Bool=true`: pre-invert the diagonal of U during LU factorisation.
    - `eltype::Type=Float64`: element type of the coefficients.

    # Examples
    ```julia
    xs = grid(32, -1, 1, GaussLobattoGrid()).xs
    D2 = DiffMatrix(xs, 7, 2; eltype = Float64)
    ```
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

"""
    size(d::DiffMatrix) -> (N, N)

Return the dense matrix dimensions represented by `d`.

`DiffMatrix` stores only `WIDTH` coefficients per row, but it behaves as a
square `N×N` matrix where `N = length(d.coeffs) ÷ WIDTH`.
"""
function Base.size(d::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    N = length(d.coeffs) ÷ WIDTH
    return (N, N)
end

"""
    IndexStyle(::DiffMatrix)

Declare Cartesian indexing for `DiffMatrix`.

The matrix is logically two-dimensional and computing a scalar entry requires
both row and column indices, so Cartesian indexing is the natural array
interface.
"""
Base.IndexStyle(::DiffMatrix) = IndexCartesian()

"""
    getindex(d::DiffMatrix, i, j)

Return the logical dense-matrix entry `d[i,j]`.

Entries outside the finite-difference stencil are returned as zero. Boundary
rows use shifted one-sided stencils, while interior rows use centered stencils.
"""
function Base.getindex(d::DiffMatrix{T, WIDTH}, i::Int, j::Int) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    # Boundary rows use shifted one-sided stencils. `offset` converts the
    # logical dense column j into the compact row-local coefficient index m.
    offset = i ≤              HWIDTH ?          HWIDTH - i + 1 :
             i > size(d, 1) - HWIDTH ? size(d, 1) - HWIDTH - i : 0
    m = HWIDTH + j - i + 1 - offset
    return (1 ≤ m ≤ WIDTH) ? d.coeffs[(i - 1)*WIDTH + m] : zero(T)
end

"""
    setindex!(d::DiffMatrix, v, i, j)

Set the stored stencil coefficient corresponding to logical entry `d[i,j]`.

If `(i,j)` lies outside the compact stencil for row `i`, the assignment is
ignored and `T(v)` is returned. This mirrors the compact storage model: values
outside the band are always represented as structural zeros.
"""
function Base.setindex!(d::DiffMatrix{T, WIDTH}, v, i::Int, j::Int) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    # Keep this index calculation in sync with getindex. Mutating entries
    # outside the stored stencil is intentionally ignored because those entries
    # are structural zeros in the compact representation.
    offset = i ≤              HWIDTH ?          HWIDTH - i + 1 :
             i > size(d, 1) - HWIDTH ? size(d, 1) - HWIDTH - i : 0
    m = HWIDTH + j - i + 1 - offset
    if 1 ≤ m ≤ WIDTH
        d.coeffs[(i - 1)*WIDTH + m] = T(v)
    end
    return T(v)
end

"""
    similar(d::DiffMatrix, [S]) -> DiffMatrix{S}

Allocate an uninitialised `DiffMatrix` with the same size, stencil width, and
optimisation flag as `d`, optionally changing the element type to `S`.
"""
function Base.similar(d::DiffMatrix{T, WIDTH, OPTIMISE}, ::Type{S}=T) where {T, S, WIDTH, OPTIMISE}
    N = length(d.coeffs) ÷ WIDTH
    return DiffMatrix{S, WIDTH, OPTIMISE}(Vector{S}(undef, N * WIDTH))
end

"""
    copy(d::DiffMatrix) -> DiffMatrix

Return a deep copy of the compact coefficient storage while preserving the
type parameters of `d`.
"""
Base.copy(d::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} =
    DiffMatrix{T, WIDTH, OPTIMISE}(copy(d.coeffs))

"""
    full(A::DiffMatrix) -> Matrix{T}

Expand the compact stencil representation into a dense `N×N` matrix.

This is mainly intended for inspection, testing, and comparison with dense
linear algebra. Applying a `DiffMatrix` with `mul!` uses the compact stencil
storage directly and avoids building this dense matrix.

# Examples
```julia
xs = range(-1, 1; length = 8)
D  = DiffMatrix(xs, 3, 1)
M  = full(D)
```
"""
function full(A::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    N   = length(A.coeffs) ÷ WIDTH
    out = zeros(T, N, N)
    @simd for i in 1:N
        # `left` is the first dense column touched by compact row i.
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

"""
    BroadcastStyle(::Type{<:DiffMatrix})

Broadcasting over a `DiffMatrix` preserves the compact finite-difference matrix
container whenever the result can still be represented with a compatible stencil
width.
"""
Base.BroadcastStyle(::Type{<:DiffMatrix{T, WIDTH, OPTIMISE}}) where {T, WIDTH, OPTIMISE} =
    DiffMatrixStyle{T, WIDTH, OPTIMISE}()

Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, s::DiffMatrixStyle) = s
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{1}, s::DiffMatrixStyle) = s
Base.BroadcastStyle(::LinearAlgebra.StructuredMatrixStyle{<:LinearAlgebra.Diagonal}, s::DiffMatrixStyle) = s
Base.BroadcastStyle(::DiffMatrixStyle{T1, W1, O1}, ::DiffMatrixStyle{T2, W2, O2}) where {T1, T2, W1, W2, O1, O2} =
    DiffMatrixStyle{promote_type(T1, T2), max(W1, W2), O1 || O2}()

"""
    similar(bc::Broadcasted{DiffMatrixStyle}, S) -> DiffMatrix{S}

Allocate the destination container for broadcasts whose dominant style is
`DiffMatrixStyle`.
"""
Base.similar(bc::Base.Broadcast.Broadcasted{DiffMatrixStyle{T, WIDTH, OPTIMISE}}, ::Type{S}) where {T, WIDTH, OPTIMISE, S} =
    DiffMatrix{S, WIDTH, OPTIMISE}(Vector{S}(undef, axes(bc)[1][end] * WIDTH))
