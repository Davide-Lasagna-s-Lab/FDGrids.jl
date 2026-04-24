# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

"""
    _make_ref(array, expr, DIM, N) -> Expr

Construct an array-indexing expression for use in generated code.

The returned AST has the form `array[i1, i2, ..., iN]` where each index is
`Symbol(:i, d)`, except along dimension `DIM`, whose index is replaced by `expr`.

# Examples
- `_make_ref(:x, :1, 2, 4)` produces `:(x[i1, 1, i3, i4])`
- `_make_ref(:x, :(i1 - 2), 1, 3)` produces `:(x[i1 - 2, i2, i3])`
"""
function _make_ref(array, expr, DIM, N)
    inds = ntuple(d -> d == DIM ? expr : Symbol(:i, d), N)
    return Expr(:ref, array, inds...)
end


# ================================================================================
# FORWARD KERNELS
# ================================================================================

"""
    _make_kernel_forward(DIM, WIDTH, N, base) -> Expr

Emit the forward stencil kernel for one grid point:

    s = Σ_{p=0}^{WIDTH-1}  A.coeffs[1+p, global_idx-1+index] · x[..., base+p, ...]
    y[..., index, ...] = s

`base` is the first stencil index; the sum is unrolled at code-generation time.
"""
function _make_kernel_forward(DIM, WIDTH, N, base)
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
    _make_loop_expr_forward(DIM, N, WIDTH) -> Expr

Construct the full nested loop for the forward operator along dimension `DIM`.
Splits the axis into head/body/tail with boundary regions of width `HWIDTH`.
"""
function _make_loop_expr_forward(DIM, N, WIDTH)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1

    # ---- preamble: boundary ranges and flags (all loop-invariant) ----
    preamble = quote
        head_range = max(1, local_rng[1]):min($HWIDTH, local_rng[end])
        body_range = max($HWIDTH + 1, local_rng[1]):min(size(x, $DIM) - $HWIDTH, local_rng[end])
        tail_range = max(size(x, $DIM) - $HWIDTH + 1, local_rng[1]):min(size(x, $DIM), local_rng[end])
        has_head   = global_idx == 1 && local_rng[1] ≤ $HWIDTH
        has_tail   = global_idx + local_rng[end] - 1 > size(A, 1) - $HWIDTH
    end

    head_kernel = _make_kernel_forward(DIM, WIDTH, N, :1)
    body_kernel = _make_kernel_forward(DIM, WIDTH, N, :($index - $HWIDTH))
    tail_kernel = _make_kernel_forward(DIM, WIDTH, N, :(size(x, $DIM) - $WIDTH + 1))

    return _wrap_loop(DIM, N, preamble, index, head_kernel, body_kernel, tail_kernel)
end


# ================================================================================
# ADJOINT KERNELS
# All three read from the precomputed transposed coefficient matrices stored in A
# (head_coeffs_T, body_coeffs_T, tail_coeffs_T), giving unit-stride column access.
# ================================================================================

"""
    _make_kernel_adjoint_head(DIM, WIDTH, N) -> Expr

Emit the transpose stencil kernel for head output points (`index ∈ 1:WIDTH`).

For head output `j`, contributing rows are `i = 1:(j+HWIDTH)` — a runtime upper
bound that grows with `j`. The kernel reads column `index` of `A.head_coeffs_T`
top-to-bottom (unit-stride in column-major storage).

This kernel only fires when `global_idx == 1`, so local and global indices coincide
and no offset is needed.
"""
function _make_kernel_adjoint_head(DIM, WIDTH, N)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1
    return quote
        s = zero(eltype(y))
        for _row in 1:($index + $HWIDTH)
            s += A.head_coeffs_T[_row, $index] * $(_make_ref(:x, :_row, DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end

"""
    _make_kernel_adjoint_body(DIM, WIDTH, N) -> Expr

Emit the transpose stencil kernel for body output points (`index ∈ WIDTH+1:N-WIDTH`).

The anti-diagonal remap is precomputed in `A.body_coeffs_T`. Column
`jb = global_idx-1+index-WIDTH` gives the WIDTH coefficients in the order
they multiply `x[index-HWIDTH], ..., x[index+HWIDTH]`. Both reads are unit-stride.
The sum is unrolled at code-generation time.
"""
function _make_kernel_adjoint_body(DIM, WIDTH, N)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1
    return quote
        # jb = global output j - WIDTH; offset by global_idx for distributed calls
        s = A.body_coeffs_T[1, global_idx - 1 + $index - $WIDTH] *
            $(_make_ref(:x, :($index - $HWIDTH), DIM, N))
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            s += A.body_coeffs_T[1 + p, global_idx - 1 + $index - $WIDTH] *
                 $(_make_ref(:x, :($index - $HWIDTH + p), DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end

"""
    _make_kernel_adjoint_tail(DIM, WIDTH, N) -> Expr

Emit the transpose stencil kernel for tail output points (`index ∈ N-WIDTH+1:N`).

For tail output `j`, contributing rows are `i = j-HWIDTH:N`. The local column
`jt` into `A.tail_coeffs_T` is computed once per output point. Reading down
column `jt` from row `jt` (unit-stride) while reading `x[index-HWIDTH:end]`
(unit-stride).

For distributed calls, `_lr` is a local x-index; `A.tail_coeffs_T` already absorbs
global row indices at adjoint-construction time, so `jt` is the only offset needed.
"""
function _make_kernel_adjoint_tail(DIM, WIDTH, N)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1
    return quote
        # jt: 1-based local column into tail_coeffs_T
        jt = global_idx - 1 + $index - size(A, 1) + $WIDTH
        s  = zero(eltype(y))
        for _lr in ($index - $HWIDTH):size(x, $DIM)
            # tail_coeffs_T row = jt + (_lr - index + HWIDTH), unit-stride as _lr increases
            s += A.tail_coeffs_T[jt + _lr - $index + $HWIDTH, jt] *
                 $(_make_ref(:x, :_lr, DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end

"""
    _make_loop_expr_adjoint(DIM, N, WIDTH) -> Expr

Construct the full nested loop for the adjoint operator along dimension `DIM`.

Boundary regions span `WIDTH` outputs on each side (wider than the forward case)
because the body kernel `body_coeffs_T` is only valid for outputs `j ∈ WIDTH+1:N-WIDTH`:
the anti-diagonal formula requires all contributing rows to be unclamped body rows.
"""
function _make_loop_expr_adjoint(DIM, N, WIDTH)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1

    # ---- preamble: wider boundary regions than forward (WIDTH, not HWIDTH) ----
    # The anti-diagonal body formula is valid only for j ∈ WIDTH+1:N-WIDTH.
    # For output j, the outermost contributing rows i=j±HWIDTH are body rows only
    # when j > WIDTH and j ≤ N-WIDTH.
    preamble = quote
        head_range = max(1, local_rng[1]):min($WIDTH, local_rng[end])
        body_range = max($WIDTH + 1, local_rng[1]):min(size(x, $DIM) - $WIDTH, local_rng[end])
        tail_range = max(size(x, $DIM) - $WIDTH + 1, local_rng[1]):min(size(x, $DIM), local_rng[end])
        has_head   = global_idx == 1 && local_rng[1] ≤ $WIDTH
        has_tail   = global_idx + local_rng[end] - 1 > size(A, 1) - $WIDTH
    end

    head_kernel = _make_kernel_adjoint_head(DIM, WIDTH, N)
    body_kernel = _make_kernel_adjoint_body(DIM, WIDTH, N)
    tail_kernel = _make_kernel_adjoint_tail(DIM, WIDTH, N)

    return _wrap_loop(DIM, N, preamble, index, head_kernel, body_kernel, tail_kernel)
end


# ================================================================================
# SHARED LOOP-WRAPPING UTILITY
# ================================================================================

"""
    _wrap_loop(DIM, N, preamble, index, head_kernel, body_kernel, tail_kernel) -> Expr

Wrap three stencil kernels in the standard nested loop structure:

1. **Phase 1** (inner loops, `d < DIM`): wrap each kernel in loops over dimensions
   below `DIM`, iterating `1:Nd` for each. `d=1` is wrapped first and ends up
   innermost, keeping `i1` as the fastest-running index (column-major).

2. **Phase 2** (stencil dimension): a block with conditional head/body/tail loops.
   `has_head` and `has_tail` are loop-invariant booleans; LLVM hoists the checks
   outside the generated loop automatically.

3. **Phase 3** (outer loops, `d > DIM`): wrap in loops over dimensions above `DIM`.

The `preamble` expression computes all ranges and flags before the loops begin.
"""
function _wrap_loop(DIM, N, preamble, index, head_kernel, body_kernel, tail_kernel)

    # phase 1: inner loops for d < DIM (d=1 wrapped first → innermost)
    inner_head = head_kernel
    inner_body = body_kernel
    inner_tail = tail_kernel
    for d in 1:DIM-1
        rng        = :(1:$(Symbol(:N, d)))
        inner_head = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_head)
        inner_body = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_body)
        inner_tail = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_tail)
    end

    # phase 2: stencil dimension block
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

    # phase 3: outer loops for d > DIM
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


# ================================================================================
# PUBLIC INTERFACE — forward DiffMatrix
# ================================================================================

"""
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}=Val(1)) -> y

Apply the forward finite-difference operator `A` to `x` along dimension `DIM`,
writing the result into `y`.

Non-distributed entry point; delegates to the lower-level method with
`global_idx=1` and `local_rng=1:size(x,DIM)`.

# Requirements
- `size(x, DIM) == size(y, DIM) == size(A.coeffs, 2)`.
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::DiffMatrix{T, WIDTH},
                            x::AbstractArray{S, N},
                             ::Val{DIM}  = Val(1)) where {T, S, N, WIDTH, DIM}
    size(x, DIM) == size(y, DIM) == size(A.coeffs, 2) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), 1, 1:size(x, DIM))
end

"""
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}, global_idx, local_rng) -> y

Distribution-aware backend for the forward operator.

`@generated` function: the loop nest, stencil unrolling, and head/body/tail
kernel selection are all specialised at compile time for the concrete
`(T, N, WIDTH, DIM)` combination.

# Arguments
- `global_idx`: Global index of `local_rng[1]` in the row numbering of `A.coeffs`.
  Set to `1` for non-distributed calls.
- `local_rng`: Local portion of dimension `DIM` to process.
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{T, N},
                                       A::DiffMatrix{TD, WIDTH},
                                       x::AbstractArray{T, N},
                                        ::Val{DIM},
                              global_idx::Int,
                               local_rng::UnitRange) where {T, TD, N, WIDTH, DIM}
    DIM in 1:N || throw(ArgumentError("inconsistent differentiation dimension"))

    block = _make_loop_expr_forward(DIM, N, WIDTH)
    Ni    = [Symbol(:N, d) for d in 1:N]

    return quote
        size(x, $DIM) == size(y, $DIM) ||
            throw(ArgumentError("inconsistent inputs size"))
        local_rng[1] > 0 && local_rng[end] ≤ size(x, $DIM) ||
            throw(ArgumentError("out of bounds local range specification"))
        global_idx > 0 && global_idx + local_rng[end] - 1 ≤ size(A, 1) ||
            throw(ArgumentError("out of bounds global/local range specification"))

        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))

        @inbounds begin
            $block
        end

        return y
    end
end


# ================================================================================
# PUBLIC INTERFACE — AdjointDiffMatrix
# ================================================================================

"""
    LinearAlgebra.mul!(y, A::AdjointDiffMatrix, x, ::Val{DIM}=Val(1)) -> y

Apply the transposed finite-difference operator `A = D*` to `x` along dimension
`DIM`, writing the result into `y`.

Uses the precomputed coefficient matrices in `A` for unit-stride access throughout.
Non-distributed entry point.

# Requirements
- `size(x, DIM) == size(y, DIM) == size(A.parent.coeffs, 2)`.
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::AdjointDiffMatrix{T, WIDTH},
                            x::AbstractArray{S, N},
                             ::Val{DIM}  = Val(1)) where {T, S, N, WIDTH, DIM}
    size(x, DIM) == size(y, DIM) == size(A.parent.coeffs, 2) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), 1, 1:size(x, DIM))
end

"""
    LinearAlgebra.mul!(y, A::AdjointDiffMatrix, x, ::Val{DIM}, global_idx, local_rng) -> y

Distribution-aware backend for the adjoint operator.

`@generated` function specialised at compile time for the concrete
`(T, N, WIDTH, DIM)` combination. The head/body/tail kernels read from
`A.head_coeffs_T`, `A.body_coeffs_T`, and `A.tail_coeffs_T` respectively —
the same unit-stride column-access pattern as the forward `mul!`.

# Arguments
- `global_idx`: Global index of `local_rng[1]`. Set to `1` for non-distributed calls.
- `local_rng`: Local portion of dimension `DIM` to process.

# Notes
For the adjoint, boundary regions span `WIDTH` outputs on each side (vs `HWIDTH`
for the forward operator) because the body formula is only valid for outputs
`j ∈ WIDTH+1:N-WIDTH`.
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{T, N},
                                       A::AdjointDiffMatrix{TD, WIDTH},
                                       x::AbstractArray{T, N},
                                        ::Val{DIM},
                              global_idx::Int,
                               local_rng::UnitRange) where {T, TD, N, WIDTH, DIM}
    DIM in 1:N || throw(ArgumentError("inconsistent differentiation dimension"))

    block = _make_loop_expr_adjoint(DIM, N, WIDTH)
    Ni    = [Symbol(:N, d) for d in 1:N]

    return quote
        size(x, $DIM) == size(y, $DIM) ||
            throw(ArgumentError("inconsistent inputs size"))
        local_rng[1] > 0 && local_rng[end] ≤ size(x, $DIM) ||
            throw(ArgumentError("out of bounds local range specification"))
        global_idx > 0 && global_idx + local_rng[end] - 1 ≤ size(A, 1) ||
            throw(ArgumentError("out of bounds global/local range specification"))

        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))

        @inbounds begin
            $block
        end

        return y
    end
end


# ================================================================================
# Point evaluation
# ================================================================================

"""
    LinearAlgebra.mul!(A::DiffMatrix{T, WIDTH}, x::AbstractVector, i::Int)

Evaluate row `i` of the forward operator `A` applied to `x`. Returns the scalar

    Σ_{p=0}^{WIDTH-1}  A.coeffs[1+p, i] * x[left+p]

where `left` is the clamped stencil start. The sum is unrolled at generation time.

# Requirements
- `length(x) == size(A, 2)`.
"""
@generated function LinearAlgebra.mul!(A::DiffMatrix{T, WIDTH},
                                       x::AbstractVector,
                                       i::Int) where {T, WIDTH}
    quote
        N = length(x)

        size(A, 2) == N || throw(DimensionMismatch())

        left = clamp(i - $WIDTH >> 1, 1, N - $WIDTH + 1)

        val = A.coeffs[1, i] * x[left]
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            val += A.coeffs[1 + p, i] * x[left + p]
        end

        return val
    end
end
