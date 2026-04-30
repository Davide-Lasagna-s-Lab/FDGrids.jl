"""
    AdjointDiffMatrix{T, WIDTH} <: AbstractMatrix{T}

Transposed finite-difference differentiation matrix D*, constructed from a
`DiffMatrix` by `adjoint(D)`.

Coefficients for the transpose operator are stored in a single flat vector
`coeffs` of length `N*WIDTH`. For each output j (1..N), the contributing
input rows are stored consecutively:

  - **head** j ∈ 1:WIDTH        → j+HWIDTH entries, inputs 1..j+HWIDTH
  - **body** j ∈ WIDTH+1:N-WIDTH → WIDTH entries,  inputs j-HWIDTH..j+HWIDTH
  - **tail** j ∈ N-WIDTH+1:N    → N-j+HWIDTH+1 entries, inputs j-HWIDTH..N

The pointer for any output j is given in closed form by `_ptr_for_j`.
"""
struct AdjointDiffMatrix{T, WIDTH} <: AbstractMatrix{T}
    parent :: DiffMatrix{T, WIDTH}
    coeffs :: Vector{T}            # length N*WIDTH, output-major order
end


# ================================================================================
# Internal coefficient builder
# ================================================================================

function _build_adjoint_coeffs(D::DiffMatrix{T, WIDTH}, w::AbstractVector{T}) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    N      = size(D, 1)
    out    = Vector{T}(undef, N * WIDTH)
    ptr    = 1
    for j in 1:WIDTH
        wj = w[j]
        for i in 1:(j + HWIDTH)
            out[ptr] = D[i, j] * w[i] / wj;  ptr += 1
        end
    end
    for j in WIDTH+1:N-WIDTH
        wj = w[j]
        for i in (j - HWIDTH):(j + HWIDTH)
            out[ptr] = D[i, j] * w[i] / wj;  ptr += 1
        end
    end
    for j in N-WIDTH+1:N
        wj = w[j]
        for i in (j - HWIDTH):N
            out[ptr] = D[i, j] * w[i] / wj;  ptr += 1
        end
    end
    return out
end


# ================================================================================
# Construction: adjoint and weighted adjoint
# ================================================================================

"""
    adjoint(D::DiffMatrix) -> AdjointDiffMatrix

Construct the transposed differentiation matrix D*.

Builds `coeffs` (length N*WIDTH) once at O(N*WIDTH) cost; all subsequent
`mul!` calls are allocation-free. Requires `size(D,1) > 2*WIDTH`.
"""
function LinearAlgebra.adjoint(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    return LinearAlgebra.adjoint(D, ones(T, size(D, 1)))
end

"""
    adjoint(D::DiffMatrix, w) -> AdjointDiffMatrix

Construct the adjoint of `D` under the weighted inner product `(u,v)_W = u'Wv`
where `W = Diagonal(w)`. The result represents D⁺ = W⁻¹ D* W.

Weights are fused into `coeffs` at construction time:
`coeff[i,j] = D[i,j] * w[i] / w[j]`. Requires `size(D,1) > 2*WIDTH`.
"""
function LinearAlgebra.adjoint(D::DiffMatrix{T, WIDTH}, w::AbstractVector{T}) where {T, WIDTH}
    size(D, 1) > 2 * WIDTH ||
        throw(ArgumentError("can only take weighted adjoint if size(D,1) > 2*WIDTH"))
    length(w) == size(D, 1) ||
        throw(ArgumentError("length(w) must equal size(D,1)"))
    return AdjointDiffMatrix(D, _build_adjoint_coeffs(D, w))
end

"""
    adjoint(D::AdjointDiffMatrix) -> DiffMatrix

Return the parent forward matrix. No copy is made.
"""
LinearAlgebra.adjoint(D::AdjointDiffMatrix) = D.parent


# ================================================================================
# AbstractMatrix interface
# ================================================================================

Base.size(d::AdjointDiffMatrix)                     = size(d.parent)
Base.IndexStyle(::AdjointDiffMatrix)                = IndexCartesian()
Base.getindex(d::AdjointDiffMatrix, i::Int, j::Int) = d.parent[j, i]

function Base.setindex!(::AdjointDiffMatrix, v, i::Int, j::Int)
    throw(ArgumentError(
        "setindex! is not supported on AdjointDiffMatrix: modifying it in-place " *
        "would leave coeffs stale. Modify the parent and reconstruct via adjoint(d.parent)."))
end

"""
    full(A::AdjointDiffMatrix) -> Matrix{T}

Expand into a dense N×N matrix equal to `transpose(full(A.parent))`.
"""
full(A::AdjointDiffMatrix) = collect(transpose(full(A.parent)))
