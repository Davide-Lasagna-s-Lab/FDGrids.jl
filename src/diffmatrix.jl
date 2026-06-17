"""
    DiffMatrix{T, WIDTH, OPTIMISE, V} <: AbstractMatrix{T}

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

# Broadcasting
Broadcasting preserves the compact `DiffMatrix` container only when the result
can still be represented by the row-local stencil storage. Supported
matrix-shaped operands are:

- compatible `DiffMatrix` objects, possibly with different stencil widths,
- `LinearAlgebra.Diagonal`,
- `LinearAlgebra.UniformScaling`, such as `I` or `3I`.

For example, `D .+ 3I` adds to the diagonal, while `D .* I` keeps only the
diagonal entries. Scalar multiplication such as `2 .* D` is also compact.
In-place broadcast assignment, for example `A .= D .+ 3I` with `A::DiffMatrix`,
writes directly into `A`'s compact coefficient storage and is intended for hot
loops. Broadcasts that would fill structural zeros, such as `D .+ 1` or `D .+
rand(size(D)...)`, are not compact operations; use `full(D)` first when a dense
result is intended.
"""
struct DiffMatrix{T, WIDTH, OPTIMISE, V<:AbstractVector{T}} <: AbstractMatrix{T}
    coeffs::V
    # `(left, right)` boundary symmetry; see `src/symmetry.jl`. This is a runtime
    # field (not a type parameter) and the main switch: a `NoSymmetry()` side
    # leaves its coefficients unchanged, while `EvenSymmetry(c)`/`OddSymmetry(c)`
    # rewrites that side's boundary rows mirrored about the centre `c`. Interior
    # rows are never changed.
    symmetry::Tuple{Symmetry, Symmetry}

    # Bypass constructor useful for CUDA adaptation (see ext/FDGridsCUDAExt.jl).
    DiffMatrix{T, WIDTH, OPTIMISE}(
        coeffs::V,
        symmetry::Tuple{Symmetry, Symmetry} = (NoSymmetry(), NoSymmetry()),
    ) where {T, WIDTH, OPTIMISE, V<:AbstractVector{T}} =
        new{T, WIDTH, OPTIMISE, V}(coeffs, symmetry)
end

"""
    DiffMatrix(xs, width, order, ::Type{T}=Float64; optimise=true,
               eltype=T, symmetry=(NoSymmetry(), NoSymmetry()))

Construct a finite-difference differentiation matrix on `xs` of the given
stencil `width` and derivative `order`.

The grid points in `xs` may be non-uniform, but they should be distinct and
ordered consistently with the arrays to which the operator will be applied.
By default the boundary rows use one-sided stencils of the same width; the
`symmetry` keyword can replace those rows with mirror stencils.

# Keyword Arguments
- `optimise::Bool=true`: pre-invert the diagonal of U during LU factorisation.
- `eltype::Type=T`: element type of the coefficients.
- `symmetry=(NoSymmetry(), NoSymmetry())`: `(left, right)` boundary symmetry,
  each a `Symmetry` object. A `NoSymmetry()` side keeps its one-sided boundary
  rows unchanged; `EvenSymmetry(c)`/`OddSymmetry(c)` rewrites that side's
  boundary rows with a stencil mirrored about the centre `c`. The centre is
  required and must be a `Real`; use `xs[1]` on the left or `xs[end]` on the
  right for boundary-centred symmetry. For an active side it must lie at or
  beyond the boundary node (`c ≤ xs[1]` on the left, `c ≥ xs[end]` on the right).
  Interior rows are never affected.
  This is a lightweight mirror stencil, not a full boundary-condition system and
  not pipe-specific regularity.

# Examples
```julia
xs = grid(32, -1, 1, GaussLobattoGrid()).xs
D2 = DiffMatrix(xs, 7, 2, Float64)
Ds = DiffMatrix(xs, 5, 1; symmetry = (EvenSymmetry(-1), NoSymmetry()))
```
"""
function DiffMatrix(xs::AbstractVector,
                 width::Int,
                 order::Int,
                      ::Type{T}=Float64;
              optimise::Bool   =true,
              eltype::Type     =T,
              symmetry::Tuple{Symmetry, Symmetry} = (NoSymmetry(), NoSymmetry())) where {T}
    3 ≤ width          || throw(ArgumentError("width must be ≥ 3"))
    width % 2 == 1     || throw(ArgumentError("width must be odd"))
    width ≤ length(xs) || throw(ArgumentError("width must not exceed the number of grid points"))

    raw = get_coeffs(xs, width, order)
    _apply_symmetry_stencil!(raw, xs, width, order, symmetry)
    coeffs = eltype.(vec(raw))

    return DiffMatrix{eltype, width, optimise}(coeffs, symmetry)
end

"""
    symmetry(D::DiffMatrix) -> (left, right)

Return the `(left, right)` tuple of boundary [`Symmetry`](@ref) objects attached
to `D`. The default for an operator built without the `symmetry` keyword is
`(NoSymmetry(), NoSymmetry())`.
"""
symmetry(D::DiffMatrix) = D.symmetry

"""
    symmetry_left(D::DiffMatrix) -> Symmetry

Return the left-boundary [`Symmetry`](@ref) of `D`, i.e. `symmetry(D)[1]`.
"""
symmetry_left(D::DiffMatrix)  = D.symmetry[1]

"""
    symmetry_right(D::DiffMatrix) -> Symmetry

Return the right-boundary [`Symmetry`](@ref) of `D`, i.e. `symmetry(D)[2]`.
"""
symmetry_right(D::DiffMatrix) = D.symmetry[2]


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
    DiffMatrix{T, WIDTH, OPTIMISE}(copy(d.coeffs), d.symmetry)

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
full(A::DiffMatrix) = [A[i, j] for i in 1:size(A, 1), j in 1:size(A, 2)]


# ================================================================================
# Broadcasting
# ================================================================================

struct DiffMatrixStyle{T, WIDTH, OPTIMISE} <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:DiffMatrix{T, WIDTH, OPTIMISE}}) where {T, WIDTH, OPTIMISE} =
    DiffMatrixStyle{T, WIDTH, OPTIMISE}()

# Compact broadcast results are supported only when every matrix-shaped operand
# can be represented in the same row-local stencil storage: scalar scaling,
# `Diagonal`, `UniformScaling`, and compatible `DiffMatrix` objects. Dense or
# sparse arbitrary matrices are deliberately not given this style; users should
# call `full(D)` first if they want dense matrix broadcasting.
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, s::DiffMatrixStyle) = s
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

const DiffMatrixBroadcasted = Base.Broadcast.Broadcasted{<:DiffMatrixStyle}
const DiffMatrixBroadcastArg = Union{DiffMatrix, DiffMatrixBroadcasted}

_unsupported_diffmatrix_broadcast(op, operand) =
    throw(ArgumentError("broadcasted `$op` between a DiffMatrix and $operand cannot be represented in compact stencil storage; use `full(D)` first for dense broadcasting"))

# Evaluate one lazy broadcast tree at one logical dense matrix index. This is
# used only for entries that have storage in the destination stencil.
@inline _bc_arg(x, I) = Base.Broadcast._broadcast_getindex(x, I)
@inline _bc_arg(J::LinearAlgebra.UniformScaling, I) = J[Tuple(I)...]
@inline _bc_arg(bc::Base.Broadcast.Broadcasted, I) = bc.f(map(x -> _bc_arg(x, I), bc.args)...)
@inline _bc_style(d::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} =
    DiffMatrixStyle{T, WIDTH, OPTIMISE}()
@inline _bc_style(::Base.Broadcast.Broadcasted{Style}) where {Style <: DiffMatrixStyle} = Style()

function Base.Broadcast.copyto!(dest::DiffMatrix{T, WIDTH}, bc::Base.Broadcast.Broadcasted{<:DiffMatrixStyle}) where {T, WIDTH}
    axes(dest) == axes(bc) || throw(DimensionMismatch("array could not be broadcast to match destination"))

    # Walk only the compact row-local stencil entries. For operations such as
    # `D .* I`, off-diagonal entries inside the stencil are written as zero; any
    # entries outside the stencil remain structural zeros because `DiffMatrix`
    # has no storage for them.
    N = size(dest, 1)
    HWIDTH = WIDTH >> 1
    @inbounds for i in 1:N
        left = clamp(i - HWIDTH, 1, N - WIDTH + 1)
        row = (i - 1) * WIDTH
        for m in 1:WIDTH
            dest.coeffs[row + m] = _bc_arg(bc, CartesianIndex(i, left + m - 1))
        end
    end
    return dest
end

# `UniformScaling` has no finite axes, so Julia's default broadcast machinery
# cannot combine it with a matrix-shaped `DiffMatrix`. Rebuild these operations
# with the `DiffMatrix` axes; `_bc_arg` then interprets `I` by indexing it.
for op in (:+, :-, :*)
    @eval begin
        Base.Broadcast.broadcasted(::typeof($op), A::DiffMatrixBroadcastArg, J::LinearAlgebra.UniformScaling) =
            Base.Broadcast.Broadcasted{typeof(_bc_style(A))}($op, (A, J), axes(A))
        Base.Broadcast.broadcasted(::typeof($op), J::LinearAlgebra.UniformScaling, A::DiffMatrixBroadcastArg) =
            Base.Broadcast.Broadcasted{typeof(_bc_style(A))}($op, (J, A), axes(A))
    end
end

for op in (:+, :-)
    @eval begin
        Base.Broadcast.broadcasted(::typeof($op), A::DiffMatrixBroadcastArg, x::Number) =
            _unsupported_diffmatrix_broadcast($op, "a scalar")
        Base.Broadcast.broadcasted(::typeof($op), x::Number, A::DiffMatrixBroadcastArg) =
            _unsupported_diffmatrix_broadcast($op, "a scalar")
    end
end

Base.Broadcast.copy(bc::Base.Broadcast.Broadcasted{<:DiffMatrixStyle}) =
    Base.Broadcast.copyto!(similar(bc, Base.Broadcast.combine_eltypes(bc.f, bc.args)), bc)
Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{<:DiffMatrixStyle}) =
    Base.Broadcast.copy(bc)
Base.Broadcast.materialize!(dest::DiffMatrix, bc::Base.Broadcast.Broadcasted{<:DiffMatrixStyle}) =
    Base.Broadcast.copyto!(dest, bc)
