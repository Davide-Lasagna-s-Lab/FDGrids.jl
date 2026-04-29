# This file defines AdjointDiffMatrix and all operations on it.
# It must be included after diffmatrix.jl (which defines DiffMatrix).

"""
    AdjointDiffMatrix{T, WIDTH} <: AbstractMatrix{T}

Transposed finite-difference differentiation matrix `D*`, constructed from a
`DiffMatrix` by `adjoint(D)`.

Rather than using Julia's generic `Adjoint` wrapper — which would require
accessing `D.coeffs` in strided anti-diagonal order at every `mul!` call —
this type precomputes three transposed coefficient matrices at construction time,
one per boundary/interior region. Every subsequent `mul!` call enjoys the same
unit-stride column-access pattern as the forward `mul!`.

# Precomputed coefficient matrices

- **`head_coeffs_T`** (`(WIDTH+HWIDTH) × WIDTH`): column `j ∈ 1:WIDTH` stores
  `D[i, j]` for contributing rows `i = 1:(j+HWIDTH)`. Entries beyond `j+HWIDTH`
  are zero and never read by the head kernel.

- **`body_coeffs_T`** (`WIDTH × (N-2*WIDTH)`): column `jb = j-WIDTH` for body
  output `j ∈ WIDTH+1:N-WIDTH`. Entry `[k, jb]` equals `D[j-HWIDTH+k-1, j]`
  (the anti-diagonal remap). Reading column `jb` top-to-bottom (unit-stride)
  multiplies against `x[j-HWIDTH:j+HWIDTH]` (unit-stride).

- **`tail_coeffs_T`** (`(WIDTH+HWIDTH) × WIDTH`): column `jt = j-(N-WIDTH)` for
  tail output `j ∈ N-WIDTH+1:N`. Entry `[k, jt]` equals `D[N-WIDTH-HWIDTH+k, j]`
  for `k ≥ jt`, zero otherwise.

# Construction cost
One-time O(N·WIDTH) allocation and fill at `adjoint(D)` call time. All subsequent
`mul!` calls are allocation-free.

# See also
`adjoint(D::DiffMatrix)`, `weighted_adjoint`, `full(D::AdjointDiffMatrix)`.
"""
struct AdjointDiffMatrix{T, WIDTH} <: AbstractMatrix{T}
    parent       ::DiffMatrix{T, WIDTH}  # original forward matrix, not copied
    head_coeffs_T::Matrix{T}             # (WIDTH+HWIDTH) × WIDTH
    body_coeffs_T::Matrix{T}             # WIDTH × (N-2*WIDTH)
    tail_coeffs_T::Matrix{T}             # (WIDTH+HWIDTH) × WIDTH
end


# ================================================================================
# Construction helpers (internal)
# ================================================================================

"""
    _build_head_coeffs_T(D) -> Matrix{T}

Build the head transposed coefficient matrix for `adjoint(D)`.

For head output `j ∈ 1:WIDTH`, contributing rows are `i = 1:j+HWIDTH`.
`head_coeffs_T[i, j] = D[i, j]`; entries beyond `j+HWIDTH` are zero.
Size: `(WIDTH+HWIDTH) × WIDTH`.
"""
function _build_head_coeffs_T(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    out    = zeros(T, WIDTH + HWIDTH, WIDTH)
    for j in 1:WIDTH
        for i in 1:(j + HWIDTH)
            out[i, j] = D[i, j]
        end
    end
    return out
end

"""
    _build_body_coeffs_T(D) -> Matrix{T}

Build the body transposed coefficient matrix for `adjoint(D)`.

For body output `j ∈ WIDTH+1:N-WIDTH`, all WIDTH contributing rows are unclamped
body rows, so the anti-diagonal formula applies:
`body_coeffs_T[k, j-WIDTH] = D[j-HWIDTH+k-1, j]` for `k = 1:WIDTH`.
Size: `WIDTH × (N-2*WIDTH)`.
"""
function _build_body_coeffs_T(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    N      = size(D, 1)
    N_body = N - 2 * WIDTH
    out    = zeros(T, WIDTH, N_body)
    for jb in 1:N_body
        j = jb + WIDTH
        for k in 1:WIDTH
            out[k, jb] = D[j - HWIDTH + k - 1, j]
        end
    end
    return out
end

"""
    _build_tail_coeffs_T(D) -> Matrix{T}

Build the tail transposed coefficient matrix for `adjoint(D)`.

For tail output `j ∈ N-WIDTH+1:N` (local column `jt = j-(N-WIDTH)`), contributing
rows are `i = j-HWIDTH:N`:
`tail_coeffs_T[k, jt] = D[N-WIDTH-HWIDTH+k, j]` for `k = jt:(WIDTH+HWIDTH)`,
zero for `k < jt`.
Size: `(WIDTH+HWIDTH) × WIDTH`.
"""
function _build_tail_coeffs_T(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    HWIDTH = WIDTH >> 1
    N      = size(D, 1)
    out    = zeros(T, WIDTH + HWIDTH, WIDTH)
    for jt in 1:WIDTH
        j = N - WIDTH + jt
        for k in jt:(WIDTH + HWIDTH)
            out[k, jt] = D[N - WIDTH - HWIDTH + k, j]
        end
    end
    return out
end


# ================================================================================
# adjoint / double-adjoint
# ================================================================================

"""
    adjoint(D::DiffMatrix) -> AdjointDiffMatrix

Construct the transposed differentiation matrix D*.

Builds the three precomputed coefficient matrices once (O(N·WIDTH)) so that all
subsequent `mul!` calls enjoy unit-stride access. Requires `size(D,1) > 2*WIDTH`
to guarantee a non-empty body region.
"""
function LinearAlgebra.adjoint(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    size(D, 1) > 2 * WIDTH ||
        throw(ArgumentError("can only take adjoint of DiffMatrix if size(D, 1) > 2*WIDTH, " *
                            "got size=$(size(D,1)), WIDTH=$WIDTH"))
    return AdjointDiffMatrix(
        D,
        _build_head_coeffs_T(D),
        _build_body_coeffs_T(D),
        _build_tail_coeffs_T(D))
end

"""
    adjoint(D::AdjointDiffMatrix) -> DiffMatrix

Return the parent forward matrix. Adjoint of adjoint is the identity; no copy
is made.
"""
LinearAlgebra.adjoint(D::AdjointDiffMatrix) = D.parent


# ================================================================================
# AbstractMatrix interface
# ================================================================================

Base.size(d::AdjointDiffMatrix) = size(d.parent)
Base.IndexStyle(::AdjointDiffMatrix) = IndexCartesian()

"""
    getindex(d::AdjointDiffMatrix, i, j)

Return element `(i, j)` of D*, i.e. `D[j, i]` of the parent forward matrix.
"""
Base.getindex(d::AdjointDiffMatrix, i::Int, j::Int) = d.parent[j, i]

"""
    setindex!(::AdjointDiffMatrix, v, i, j)

Not supported. Modifying an `AdjointDiffMatrix` in-place would leave its
precomputed coefficient matrices stale. Modify the parent and reconstruct:

```julia
d.parent[j, i] = v
d = adjoint(d.parent)
```
"""
function Base.setindex!(::AdjointDiffMatrix, v, i::Int, j::Int)
    throw(ArgumentError(
        "setindex! is not supported on AdjointDiffMatrix: modifying it in-place " *
        "would leave the precomputed coefficient matrices stale.\n" *
        "Modify the parent DiffMatrix and reconstruct via adjoint(d.parent)."))
end

"""
    full(A::AdjointDiffMatrix) -> Matrix{T}

Expand the transposed representation into a dense `N×N` matrix,
equal to `transpose(full(A.parent))`.
"""
full(A::AdjointDiffMatrix) = collect(transpose(full(A.parent)))


# ================================================================================
# Weighted adjoint:  D⁺ = W⁻¹ D* W
# ================================================================================

"""
    weighted_adjoint(D::DiffMatrix{T, WIDTH}, w::AbstractVector{T}) -> AdjointDiffMatrix{T, WIDTH}

Construct the adjoint of `D` under the weighted inner product `(u, v)_W = u'Wv`
where `W = Diagonal(w)`. The result represents the operator

    D⁺ = W⁻¹ D* W

whose action on a vector `v` is `(1/w) .* (D* (w .* v))`.

The weights are fused directly into the three precomputed coefficient matrices at
construction time, so each `mul!(y, Dp, x)` call has the same cost as an unweighted
`mul!(y, adjoint(D), x)` — no additional scaling passes are needed.

The fused coefficient for output `j` and contributing row `i` is:

    coeff_fused[i, j] = D[i, j] * w[i] / w[j]

# Arguments
- `D`: Forward differentiation matrix.
- `w`: Weight vector of length `size(D, 1)`. All entries must be non-zero.

# Returns
An `AdjointDiffMatrix` with scaled coefficient matrices.

# See also
`adjoint(D::DiffMatrix)` for the unweighted transpose.
"""
function weighted_adjoint(D::DiffMatrix{T, WIDTH}, w::AbstractVector{T}) where {T, WIDTH}
    size(D, 1) > 2 * WIDTH ||
        throw(ArgumentError("can only take weighted adjoint if size(D, 1) > 2*WIDTH"))
    length(w) == size(D, 1) ||
        throw(ArgumentError("length(w) must equal size(D, 1)"))

    HWIDTH = WIDTH >> 1
    N      = size(D, 1)

    # build the base transposed coefficient matrices, then scale in-place
    head = _build_head_coeffs_T(D)
    body = _build_body_coeffs_T(D)
    tail = _build_tail_coeffs_T(D)

    # head: head[i, j] *= w[i] / w[j]  for i = 1:j+HWIDTH, j = 1:WIDTH
    for j in 1:WIDTH
        for i in 1:(j + HWIDTH)
            head[i, j] *= w[i] / w[j]
        end
    end

    # body: body[k, jb] *= w[j-HWIDTH+k-1] / w[j]  where j = jb+WIDTH
    for jb in 1:size(body, 2)
        j = jb + WIDTH
        for k in 1:WIDTH
            body[k, jb] *= w[j - HWIDTH + k - 1] / w[j]
        end
    end

    # tail: tail[k, jt] *= w[N-WIDTH-HWIDTH+k] / w[j]  where j = N-WIDTH+jt
    for jt in 1:WIDTH
        j = N - WIDTH + jt
        for k in jt:(WIDTH + HWIDTH)
            tail[k, jt] *= w[N - WIDTH - HWIDTH + k] / w[j]
        end
    end

    return AdjointDiffMatrix(D, head, body, tail)
end
