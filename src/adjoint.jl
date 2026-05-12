"""
    AdjointDiffMatrix{T, WIDTH} <: AbstractMatrix{T}

Transposed finite-difference differentiation matrix D*, constructed from a
`DiffMatrix` by `adjoint(D)`.

`adjoint(D, w)` constructs the adjoint under the weighted inner product with
diagonal weights `w`, representing `W^-1 * transpose(D) * W`.

The parent `DiffMatrix` is retained so `adjoint(A::AdjointDiffMatrix)` can
return the original operator without copying. For weighted adjoints this is a
structural unwrap, not the standard Euclidean adjoint of the weighted operator.

Coefficients for the transpose operator are stored in a single flat vector
`coeffs` of length `N*WIDTH`. For each output j (1..N), the contributing
input rows are stored consecutively:

  - **head** j ∈ 1:WIDTH        → j+HWIDTH entries, inputs 1..j+HWIDTH
  - **body** j ∈ WIDTH+1:N-WIDTH → WIDTH entries,  inputs j-HWIDTH..j+HWIDTH
  - **tail** j ∈ N-WIDTH+1:N    → N-j+HWIDTH+1 entries, inputs j-HWIDTH..N

The pointer for any output j is given in closed form by `_ptr_for_j`.

# Examples
```julia
using LinearAlgebra

xs = range(-1, 1; length = 16)
D  = DiffMatrix(xs, 5, 1)
Dt = adjoint(D)

y = similar(collect(xs))
mul!(y, Dt, sin.(xs))
```
"""
struct AdjointDiffMatrix{T, WIDTH} <: AbstractMatrix{T}
    parent :: DiffMatrix{T, WIDTH}
    coeffs :: Vector{T}            # length N*WIDTH, output-major order
end


# ================================================================================
# Internal coefficient layout
# ================================================================================

"""
    _ptr_for_j(j, N, ::Val{WIDTH}) -> Int

Return the 1-based index into `A.coeffs` of the first coefficient for output `j`.

This is used by both scalar indexing and generated `mul!` kernels. It encodes
the three-region storage layout used by `AdjointDiffMatrix`: a growing head,
a fixed-width body, and a shrinking tail.

  - head  j ≤ WIDTH:           1 + HWIDTH*(j-1) + (j-1)*j÷2
  - body  WIDTH < j ≤ N-WIDTH: (j-1)*WIDTH + 1
  - tail  j > N-WIDTH:         (N-WIDTH)*WIDTH + 1 + (WIDTH+HWIDTH+1)*(jt-1) - (jt-1)*jt÷2
                                where jt = j-(N-WIDTH)
"""
function _ptr_for_j(j::Int, N::Int, ::Val{WIDTH}) where {WIDTH}
    HWIDTH = WIDTH >> 1
    if j ≤ WIDTH
        # Head rows have lengths HWIDTH+1, HWIDTH+2, ..., HWIDTH+j.
        # Summing those previous row lengths gives the triangular term.
        return 1 + HWIDTH*(j - 1) + (j - 1)*j÷2
    elseif j ≤ N - WIDTH
        # Body rows all have fixed length WIDTH. The head region stores exactly
        # WIDTH^2 coefficients, so this collapses to the same row-major formula.
        return (j - 1)*WIDTH + 1
    else
        # Tail rows shrink by one coefficient per row. Count from the first tail
        # row, after all head and body coefficients have been stored.
        jt = j - (N - WIDTH)
        return (N - WIDTH)*WIDTH + 1 +
               (WIDTH + HWIDTH + 1)*(jt - 1) - (jt - 1)*jt÷2
    end
end

"""
    _build_adjoint_coeffs(D, w) -> Vector

Build the compact coefficient vector for the weighted adjoint of `D`.

For a weighted inner product with diagonal weights `w`, the adjoint operator is
`W⁻¹ D' W`. This helper folds the factor `w[i] / w[j]` into each stored
coefficient so later `mul!` calls only perform the stencil dot products.
"""
function _build_adjoint_coeffs(D::DiffMatrix{T, WIDTH}, w::AbstractVector{T}) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    N      = size(D, 1)
    out    = Vector{T}(undef, N * WIDTH)
    ptr    = 1

    # The loops are over adjoint output rows j, i.e. columns of the parent D.
    # For each nonzero D[i,j], store the coefficient used in
    #     y[j] += D[i,j] * w[i] / w[j] * x[i].
    # With w == ones(T, N), this is the ordinary transpose.

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

# Examples
```julia
using LinearAlgebra

xs = range(-1, 1; length = 16)
D  = DiffMatrix(xs, 5, 1)
Dt = adjoint(D)
```
"""
function LinearAlgebra.adjoint(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    return LinearAlgebra.adjoint(D, ones(T, size(D, 1)))
end

"""
    adjoint(D::DiffMatrix, w) -> AdjointDiffMatrix

Construct the adjoint of `D` under the weighted inner product `(u,v)_W = u'Wv`
where `W = Diagonal(w)`. The result represents D⁺ = W⁻¹ D* W.

Weights are fused into `coeffs` at construction time:
the stored entry for adjoint row `j` and source row `i` is
`D[i,j] * w[i] / w[j]`. Requires `size(D,1) > 2*WIDTH`.

# Examples
```julia
using LinearAlgebra

g  = grid(32, -1, 1, GaussLobattoGrid())
D  = DiffMatrix(g.xs, 5, 1)
Dt = adjoint(D, g.ws)
```
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

This is a structural operation: `AdjointDiffMatrix` stores a reference to the
`DiffMatrix` from which it was constructed, and this method returns that object.
For an unweighted adjoint this agrees with taking the adjoint twice. For a
weighted adjoint built by `adjoint(D, w)`, this does **not** mean that the
standard Euclidean adjoint of `W^-1 * transpose(D) * W` is `D`; the weights are
not stored separately, and this method intentionally discards the weighted
operator view.
"""
LinearAlgebra.adjoint(D::AdjointDiffMatrix) = D.parent


# ================================================================================
# AbstractMatrix interface
# ================================================================================

"""
    size(d::AdjointDiffMatrix) -> (N, N)

Return the dimensions of the adjoint operator. The adjoint has the same logical
size as its parent `DiffMatrix`.
"""
Base.size(d::AdjointDiffMatrix)                     = size(d.parent)

"""
    IndexStyle(::AdjointDiffMatrix)

Declare Cartesian indexing for the logical matrix interface.
"""
Base.IndexStyle(::AdjointDiffMatrix)                = IndexCartesian()

"""
    getindex(d::AdjointDiffMatrix, i, j)

Return the dense logical entry of the adjoint operator.

This method reads from the precomputed adjoint coefficient storage, so it also
reflects weighted adjoints constructed by `adjoint(D, w)`. Performance-critical
application should use `mul!`, which avoids scalar indexing overhead.
"""
function Base.getindex(d::AdjointDiffMatrix{T, WIDTH}, i::Int, j::Int) where {T, WIDTH}
    checkbounds(d, i, j)

    N      = size(d, 1)
    HWIDTH = WIDTH >> 1

    # Row i of the adjoint has variable support near the boundaries. These
    # bounds mirror the head/body/tail layout used in _build_adjoint_coeffs.
    ilo    = i ≤ WIDTH ? 1 :
             i ≤ N - WIDTH ? i - HWIDTH :
             i - HWIDTH
    ihi    = i ≤ WIDTH ? i + HWIDTH :
             i ≤ N - WIDTH ? i + HWIDTH :
             N

    ilo ≤ j ≤ ihi || return zero(T)
    return d.coeffs[_ptr_for_j(i, N, Val(WIDTH)) + j - ilo]
end

"""
    setindex!(d::AdjointDiffMatrix, v, i, j)

Reject mutation of adjoint entries.

`AdjointDiffMatrix` stores precomputed coefficients derived from its parent.
Allowing scalar mutation would make those coefficients inconsistent, so callers
should modify the parent matrix and reconstruct the adjoint instead.
"""
function Base.setindex!(::AdjointDiffMatrix, v, i::Int, j::Int)
    throw(ArgumentError(
        "setindex! is not supported on AdjointDiffMatrix: modifying it in-place " *
        "would leave coeffs stale. Modify the parent and reconstruct via adjoint(d.parent)."))
end

"""
    full(A::AdjointDiffMatrix) -> Matrix{T}

Expand the adjoint operator into a dense N×N matrix.

For weighted adjoints, this expands the weighted operator represented by
`A.coeffs`, not merely the unweighted transpose of the parent matrix.
"""
full(A::AdjointDiffMatrix{T}) where {T} = [A[i, j] for i in 1:size(A, 1), j in 1:size(A, 2)]
