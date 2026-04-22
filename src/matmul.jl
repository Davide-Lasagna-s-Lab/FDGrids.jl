# ----------------
# HELPER FUNCTIONS
# ----------------

"""
    _make_ref(array, expr, DIM, N) -> Expr

Construct an array-indexing expression for use in generated code.

The returned AST has the form

    array[i1, i2, ..., iN]

where each index is the loop variable `Symbol(:i, d)`, except along dimension
`DIM`, whose index is replaced by `expr`. This is a small utility used when
building the generated stencil kernels.

# Arguments
- `array`: Symbol naming the array to index, such as `:x` or `:y`.
- `expr`: Expression to place in the `DIM`-th index position.
- `DIM`: Target dimension, using 1-based indexing.
- `N`: Total number of array dimensions.

# Returns
- An `Expr(:ref, ...)` representing the indexed array access.

# Examples
- `_make_ref(:x, :1, 2, 4)` produces `:(x[i1, 1, i3, i4])`
- `_make_ref(:x, :(base + p), 3, 4)` produces `:(x[i1, i2, base + p, i4])`

# Notes
This function does not validate `DIM` or `N`; callers are expected to pass
consistent values.
"""
function _make_ref(array, expr, DIM, N)
    inds = ntuple(d -> d == DIM ? expr : Symbol(:i, d), N)
    return Expr(:ref, array, inds...)
end


"""
    _make_kernel_expr(DIM, WIDTH, N, base, ::Val{false}) -> Expr

Build the forward stencil kernel for one grid point. The emitted code evaluates

    s = Σ_{p=0}^{WIDTH-1}  A.coeffs[1+p, global_idx-1+index] · x[..., base+p, ...]
    y[..., index, ...] = s

where `index = Symbol(:i, DIM)` is the loop variable along the differentiated
dimension. The summation is unrolled at code-generation time.

# Arguments
- `DIM`: Differentiation dimension.
- `WIDTH`: Number of stencil points.
- `N`: Total number of dimensions of `x` and `y`.
- `base`: Expression for the first index of the stencil window along `DIM`.
  Typical values are `:1` (head), `:(index - HWIDTH)` (body), or
  `:(size(x, DIM) - WIDTH + 1)` (tail).

# Expected runtime bindings
`A`, `x`, `y`, and `global_idx` must be in scope in the emitted code.
`global_idx` is `1` in the non-distributed case and greater than `1` when
the operator is applied to a local slice of a globally indexed dimension.
"""
function _make_kernel_expr(DIM, WIDTH, N, base, ::Val{false})
    index = Symbol(:i, DIM)
    return quote
        s = A.coeffs[1, global_idx - 1 + $index] * $(_make_ref(:x, base, DIM, N))
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            s += A.coeffs[1 + p, global_idx - 1 + $index] *
                 $(_make_ref(:x, :(($base) + p), DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end


"""
    _make_kernel_expr(DIM, WIDTH, N, base, ::Val{true}) -> Expr

Build the transpose stencil kernel for one body grid point.

In the body region, `left(i) = i - HWIDTH` is unclamped, so the coefficient
slot for `A^T[index, i]` follows the clean anti-diagonal formula:

    slot = WIDTH - p   for row  i = base + p

This gives

    s = Σ_{p=0}^{WIDTH-1}  A.coeffs[WIDTH-p, base+p] · x[..., base+p, ...]
    y[..., index, ...] = s

where `base = index - HWIDTH`.

This kernel is correct only in the body region. For head and tail use
`_make_kernel_transpose_head` and `_make_kernel_transpose_tail` respectively,
where the clamped `left(i)` changes the coefficient slot formula.
"""
function _make_kernel_expr(DIM, WIDTH, N, base, ::Val{true})
    index = Symbol(:i, DIM)
    return quote
        # A.coeffs columns are global: offset local base by global_idx - 1.
        # The x accesses use local base — the stencil window is the same in both cases.
        s = A.coeffs[$WIDTH, global_idx - 1 + $base] * $(_make_ref(:x, base, DIM, N))
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            s += A.coeffs[$WIDTH - p, global_idx - 1 + ($base) + p] *
                 $(_make_ref(:x, :(($base) + p), DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end


"""
    _make_kernel_transpose_head(DIM, WIDTH, N) -> Expr

Build the transpose stencil kernel for head output points (`index ∈ 1:WIDTH`).

For head output j, the contributing rows are all i such that D[i,j] ≠ 0. In the
head region, this includes head-clamped rows (left=1) and body rows whose stencil
window reaches back to j. The full set is i ∈ 1:(j+HWIDTH), which grows with j —
so the loop bound is runtime rather than a compile-time constant.

`getindex` computes the correct coefficient for any (row, col) pair, handling
clamped and unclamped rows transparently, and returning zero outside the stencil.
This kernel only fires when `global_idx == 1` (first process), so local indices
equal global indices and no offset is needed.
"""
function _make_kernel_transpose_head(DIM, WIDTH, N)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1
    return quote
        s = zero(eltype(y))
        for _row in 1:($index + $HWIDTH)
            s += A[_row, $index] * $(_make_ref(:x, :_row, DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end


"""
    _make_kernel_transpose_tail(DIM, WIDTH, N) -> Expr

Build the transpose stencil kernel for tail output points (`index ∈ N-WIDTH+1:N`).

For tail output j, the contributing rows are all i such that D[i,j] ≠ 0. This
includes tail-clamped rows (left=N-WIDTH+1) and body rows whose stencil window
reaches forward to j. The full set is i ∈ (j-HWIDTH):N — the lower bound shrinks
as j decreases toward N-WIDTH+1, so the loop bound is runtime rather than a
compile-time constant. This is exactly the symmetric counterpart of
`_make_kernel_transpose_head`.

`getindex` computes the correct coefficient for any (row, col) pair. For the
distributed case, `_row` is a local index into x, while A is addressed with
global coordinates (`global_idx - 1` offsets both row and column).
"""
function _make_kernel_transpose_tail(DIM, WIDTH, N)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1
    return quote
        s = zero(eltype(y))
        # _row is a local index; convert to global for A, keep local for x
        for _row in ($index - $HWIDTH):size(x, $DIM)
            s += A[global_idx - 1 + _row, global_idx - 1 + $index] *
                 $(_make_ref(:x, :_row, DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end


"""
    _make_loop_expr(DIM, N, WIDTH, IS_ADJOINT) -> Expr

Construct the full nested loop expression that applies the finite-difference
stencil (or its transpose) along dimension `DIM`.

The generated code splits the differentiated axis into three regions:

- **head**: points near the left boundary, where the stencil is pinned to the
  start of the domain.
- **body**: interior points, where the stencil is centred.
- **tail**: points near the right boundary, where the stencil is pinned to the
  end of the domain.

For the forward operator (`IS_ADJOINT = false`), all three kernels use
`_make_kernel_expr(..., Val(false))` with the appropriate pinned `base`.

For the transpose operator (`IS_ADJOINT = true`), the body uses the clean
anti-diagonal formula via `_make_kernel_expr(..., Val(true))`, while the head
and tail use `_make_kernel_transpose_head` and `_make_kernel_transpose_tail`
respectively, which handle the clamped `left(i)` formula correctly without
any runtime dispatch into `getindex`.

# Arguments
- `DIM`: Differentiation dimension.
- `N`: Number of array dimensions.
- `WIDTH`: Stencil width (expected to be odd).
- `IS_ADJOINT`: `Bool` compile-time flag; `true` selects the transpose kernels.

# Structure of the generated code
1. A preamble computes `head_range`, `body_range`, `tail_range`, and the flags
   `has_head` and `has_tail`. All of these are loop-invariant; LLVM hoists the
   flag checks outside the loop automatically.
2. Three stencil kernels are selected based on `IS_ADJOINT`.
3. The kernels are wrapped in loops over all dimensions so that dimension 1
   is the fastest-running index (column-major layout).

# Expected runtime bindings
`A`, `x`, `y`, `global_idx`, `local_rng`, and `N1, N2, ..., NN` must be in
scope in the emitted code.
"""
function _make_loop_expr(DIM, N, WIDTH, IS_ADJOINT)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1

    # ---- preamble: compute ranges and boundary flags at runtime ----
    # These are all loop-invariant; LLVM will hoist the has_head/has_tail
    # checks outside the generated loop automatically.
    #
    # Forward: boundary regions span HWIDTH outputs each side — exactly where
    # the stencil is pinned rather than centred.
    #
    # Transpose: boundary regions must span WIDTH outputs each side. The
    # anti-diagonal body formula A.coeffs[WIDTH-p, base+p] is only valid when
    # every contributing row i is a pure body row (left(i) = i-HWIDTH unclamped).
    # For output j, the contributing rows are i ∈ j-HWIDTH:j+HWIDTH. Row i is
    # a body row only when i > HWIDTH and i ≤ N-HWIDTH. The most restrictive
    # condition is i=j-HWIDTH > HWIDTH → j > WIDTH, and i=j+HWIDTH ≤ N-HWIDTH
    # → j ≤ N-WIDTH. So the body formula is safe only for j ∈ WIDTH+1:N-WIDTH.
    preamble = if IS_ADJOINT
        quote
            head_range = max(1, local_rng[1]):min($WIDTH, local_rng[end])
            body_range = max($WIDTH + 1, local_rng[1]):min(size(x, $DIM) - $WIDTH, local_rng[end])
            tail_range = max(size(x, $DIM) - $WIDTH + 1, local_rng[1]):min(size(x, $DIM), local_rng[end])
            has_head   = global_idx == 1 && local_rng[1] ≤ $WIDTH
            has_tail   = global_idx + local_rng[end] - 1 > size(A, 1) - $WIDTH
        end
    else
        quote
            head_range = max(1, local_rng[1]):min($HWIDTH, local_rng[end])
            body_range = max($HWIDTH + 1, local_rng[1]):min(size(x, $DIM) - $HWIDTH, local_rng[end])
            tail_range = max(size(x, $DIM) - $HWIDTH + 1, local_rng[1]):min(size(x, $DIM), local_rng[end])
            has_head   = global_idx == 1 && local_rng[1] ≤ $HWIDTH
            has_tail   = global_idx + local_rng[end] - 1 > size(A, 1) - $HWIDTH
        end
    end

    # ---- kernels: one per region, selected at code-generation time ----
    if IS_ADJOINT
        head_kernel = _make_kernel_transpose_head(DIM, WIDTH, N)
        body_kernel = _make_kernel_expr(DIM, WIDTH, N, :($index - $HWIDTH), Val(true))
        tail_kernel = _make_kernel_transpose_tail(DIM, WIDTH, N)
    else
        head_kernel = _make_kernel_expr(DIM, WIDTH, N, :1,                            Val(false))
        body_kernel = _make_kernel_expr(DIM, WIDTH, N, :($index - $HWIDTH),           Val(false))
        tail_kernel = _make_kernel_expr(DIM, WIDTH, N, :(size(x, $DIM) - $WIDTH + 1), Val(false))
    end

    # ---- phase 1: wrap kernels in inner loops for d < DIM ----
    # Iterating 1:DIM-1 means d=1 is wrapped first and ends up innermost,
    # keeping i1 as the fastest-running index (column-major order).
    inner_head = head_kernel
    inner_body = body_kernel
    inner_tail = tail_kernel
    for d in 1:DIM-1
        rng        = :(1:$(Symbol(:N, d)))
        inner_head = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_head)
        inner_body = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_body)
        inner_tail = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_tail)
    end

    # ---- phase 2: stencil dimension block ----
    # head and tail are guarded by loop-invariant flags; body always runs.
    dim_block = quote
        if has_head
            for $index in head_range
                $inner_head
            end
        end
        for $index in body_range
            $inner_body
        end
        if has_tail
            for $index in tail_range
                $inner_tail
            end
        end
    end

    # ---- phase 3: wrap in outer loops for d > DIM ----
    loop = dim_block
    for d in DIM+1:N
        rng  = :(1:$(Symbol(:N, d)))
        loop = Expr(:for, Expr(:(=), Symbol(:i, d), rng), loop)
    end

    return quote
        $preamble
        $loop
    end
end


# ----------------
# PUBLIC INTERFACE
# ----------------

"""
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}=Val(1)) -> y

Apply the finite-difference operator `A` to `x` along dimension `DIM`, writing
the result into `y`.

This is the standard non-distributed entry point. It delegates to the
lower-level method with `global_idx = 1` and `local_rng = 1:size(x, DIM)`.

# Requirements
- `size(x, DIM) == size(y, DIM) == size(A.coeffs, 2)`.
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::DiffMatrix{T, WIDTH},
                            x::AbstractArray{S, N},
                             ::Val{DIM}=Val(1)) where {T, S, N, WIDTH, DIM}
    size(x, DIM) == size(y, DIM) == size(A.coeffs, 2) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), Val(false), 1, 1:size(x, DIM))
end


"""
    LinearAlgebra.mul!(y, At::Adjoint{<:DiffMatrix}, x, ::Val{DIM}=Val(1)) -> y

Apply the transpose of the finite-difference operator to `x` along dimension
`DIM`, writing the result into `y`.

`At.parent` is extracted immediately so that the concrete `DiffMatrix` type
flows into the lower-level `@generated` method without any Union dispatch.

# Requirements
- `size(x, DIM) == size(y, DIM) == size(At.parent.coeffs, 2)`.
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                           At::LinearAlgebra.Adjoint{<:Any, <:DiffMatrix{T, WIDTH}},
                            x::AbstractArray{S, N},
                             ::Val{DIM}=Val(1)) where {T, S, N, WIDTH, DIM}
    A = At.parent
    size(x, DIM) == size(y, DIM) == size(A.coeffs, 2) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), Val(true), 1, 1:size(x, DIM))
end


"""
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}, ::Val{IS_ADJOINT},
                       global_idx, local_rng) -> y

Distribution-aware backend for applying `A` (or its transpose) along dimension
`DIM` over a local subrange.

This is a `@generated` function: the loop nest, stencil unrolling, and
head/body/tail kernel selection are all fully specialised at compile time for
the concrete `(T, N, WIDTH, DIM, IS_ADJOINT)` combination.

# Arguments
- `y`: Destination array.
- `A`: Finite-difference operator (always the forward `DiffMatrix`, even for
  the transpose case — the caller extracts `At.parent` before calling this).
- `x`: Source array.
- `Val(DIM)`: Differentiation dimension, known at compile time.
- `Val(IS_ADJOINT)`: `Val(false)` for forward, `Val(true)` for transpose.
- `global_idx::Int`: Global index of `local_rng[1]` in the row numbering of
  `A.coeffs`. Set to `1` for non-distributed calls.
- `local_rng::UnitRange`: Local portion of dimension `DIM` to process.

# Notes
Do not call this method directly unless you need distributed or blockwise
application. Use the higher-level `mul!(y, A, x, Val(DIM))` entry points.
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{T, N},
                                       A::DiffMatrix{TD, WIDTH},
                                       x::AbstractArray{T, N},
                                        ::Val{DIM},
                                        ::Val{IS_ADJOINT},
                              global_idx::Int,
                               local_rng::UnitRange) where {T, TD, N, WIDTH, DIM, IS_ADJOINT}
    # compile-time safety check
    DIM in 1:N || throw(ArgumentError("inconsistent differentiation dimension"))

    # build the full nested loop expression at compile time
    block = _make_loop_expr(DIM, N, WIDTH, IS_ADJOINT)

    # symbols for the sizes of each dimension, e.g. N1, N2, N3
    Ni = [Symbol(:N, d) for d in 1:N]

    return quote
        # runtime sanity checks
        size(x, $DIM) == size(y, $DIM) ||
            throw(ArgumentError("inconsistent inputs size"))
        local_rng[1] > 0 && local_rng[end] ≤ size(x, $DIM) ||
            throw(ArgumentError("out of bounds local range specification"))
        global_idx > 0 && global_idx + local_rng[end] - 1 ≤ size(A, 1) ||
            throw(ArgumentError("out of bounds global/local range specification"))

        # destructure size(y) into N1, N2, ... for use in the generated loops
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))

        @inbounds begin
            $block
        end

        return y
    end
end


"""
    LinearAlgebra.mul!(A::DiffMatrix{T, WIDTH}, x::AbstractVector, i::Int)

Evaluate row `i` of the finite-difference operator `A` applied to `x`.

Returns the scalar

    sum(A.coeffs[1+p, i] * x[left+p] for p = 0:WIDTH-1)

where `left` is the clamped stencil start index. The stencil sum is unrolled
at generation time.

# Requirements
- `length(x) == size(A, 2)`.
"""
@generated function LinearAlgebra.mul!(A::DiffMatrix{T, WIDTH},
                                       x::AbstractVector,
                                       i::Int) where {T, WIDTH}
    quote
        N = length(x)

        size(A, 2) == N || throw(DimensionMismatch())

        # index of the first element of the stencil
        left = clamp(i - $WIDTH >> 1, 1, N - $WIDTH + 1)

        val = A.coeffs[1, i] * x[left]
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            val += A.coeffs[1 + p, i] * x[left + p]
        end

        return val
    end
end
