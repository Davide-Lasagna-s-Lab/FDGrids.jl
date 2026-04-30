# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

"""
    _make_ref(array, expr, DIM, N) -> Expr

Construct an array-indexing expression for use in generated code.

The returned AST has the form `array[i1, i2, ..., iN]` where each index is
`Symbol(:i, d)`, except along dimension `DIM`, whose index is replaced by `expr`.
"""
function _make_ref(array, expr, DIM, N)
    inds = ntuple(d -> d == DIM ? expr : Symbol(:i, d), N)
    return Expr(:ref, array, inds...)
end


# ================================================================================
# KERNEL BUILDERS
# ================================================================================

"""
    _make_kernel_fixed(DIM, WIDTH, N, coeff_field, base_expr) -> Expr

Emit a stencil kernel that reads exactly WIDTH coefficients from
`A.<coeff_field>[ptr..]` and WIDTH x-values starting at `x[base_expr..]`.
The dot-product is fully unrolled at code-generation time.

Covers all forward regions and the adjoint body:
  - forward head:    coeff_field=:coeffs,   base_expr=1
  - forward body:    coeff_field=:coeffs,   base_expr=:(index - HWIDTH)
  - forward tail:    coeff_field=:coeffs,   base_expr=:(size(x,DIM) - WIDTH + 1)
  - adjoint body:    coeff_field=:coeffs_T, base_expr=:(index - HWIDTH)
"""
function _make_kernel_fixed(DIM, WIDTH, N, coeff_field::Symbol, base_expr)
    index = Symbol(:i, DIM)
    return quote
        s = A.$coeff_field[ptr] * $(_make_ref(:x, base_expr, DIM, N))
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            s += A.$coeff_field[ptr + p] * $(_make_ref(:x, :(($base_expr) + p), DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end

"""
    _make_kernel_variable(DIM, WIDTH, N, nrows_expr, base_expr) -> Expr

Emit a stencil kernel that reads a variable number of terms from
`A.coeffs_T[ptr..]` and `x[base_expr + _k - 1]` for `_k in 1:nrows_j`.

Used for adjoint boundary outputs whose stencil row-count varies:
  - adjoint head: nrows_expr=:(index + HWIDTH),              base_expr=1
  - adjoint tail: nrows_expr=:(size(x,DIM) - index + HWIDTH + 1), base_expr=:(index - HWIDTH)
"""
function _make_kernel_variable(DIM, WIDTH, N, nrows_expr, base_expr)
    index = Symbol(:i, DIM)
    return quote
        nrows_j = $nrows_expr
        s = zero(eltype(y))
        for _k in 1:nrows_j
            s += A.coeffs_T[ptr + _k - 1] * $(_make_ref(:x, :(($base_expr) + _k - 1), DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end


# ================================================================================
# LOOP BUILDERS — forward and adjoint
# ================================================================================

"""
    _make_loop_expr_forward(DIM, N, WIDTH) -> Expr

Construct the full nested loop for the forward operator along dimension `DIM`.
Uses a running `ptr` into `A.coeffs` that advances by `WIDTH` per output.
Each fiber (outer-loop iteration) resets `ptr = ptr_start`.
"""
function _make_loop_expr_forward(DIM, N, WIDTH)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1

    preamble = quote
        head_range = max(1, local_rng[1]):min($HWIDTH, local_rng[end])
        body_range = max($HWIDTH + 1, local_rng[1]):min(size(x, $DIM) - $HWIDTH, local_rng[end])
        tail_range = max(size(x, $DIM) - $HWIDTH + 1, local_rng[1]):min(size(x, $DIM), local_rng[end])
        has_head   = global_idx == 1 && local_rng[1] ≤ $HWIDTH
        has_tail   = global_idx + local_rng[end] - 1 > size(A, 1) - $HWIDTH
        _f_local   = has_head ? local_rng[1] :
                     !isempty(body_range) ? first(body_range) :
                     first(tail_range)
        ptr_start  = (global_idx + _f_local - 2) * $WIDTH + 1
    end

    head_kernel = _make_kernel_fixed(DIM, WIDTH, N, :coeffs, 1)
    body_kernel = _make_kernel_fixed(DIM, WIDTH, N, :coeffs, :($index - $HWIDTH))
    tail_kernel = _make_kernel_fixed(DIM, WIDTH, N, :coeffs, :(size(x, $DIM) - $WIDTH + 1))

    inner_head = head_kernel
    inner_body = body_kernel
    inner_tail = tail_kernel
    for d in 1:DIM-1
        rng        = :(1:$(Symbol(:N, d)))
        inner_head = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_head)
        inner_body = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_body)
        inner_tail = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_tail)
    end

    advance = :(ptr += $WIDTH)

    dim_block = quote
        ptr = ptr_start
        if has_head
            for $index in head_range
                $inner_head
                $advance
            end
        end
        for $index in body_range
            $inner_body
            $advance
        end
        if has_tail
            for $index in tail_range
                $inner_tail
                $advance
            end
        end
    end

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
# ADJOINT HELPER
# ================================================================================

"""
    _ptr_for_j(j, N, ::Val{WIDTH}) -> Int

Return the 1-based index into `coeffs_T` of the first coefficient for output `j`.

  - head  j ≤ WIDTH:           1 + HWIDTH*(j-1) + (j-1)*j÷2
  - body  WIDTH < j ≤ N-WIDTH: (j-1)*WIDTH + 1
  - tail  j > N-WIDTH:         (N-WIDTH)*WIDTH + 1 + (WIDTH+HWIDTH+1)*(jt-1) - (jt-1)*jt÷2
                                where jt = j-(N-WIDTH)
"""
function _ptr_for_j(j::Int, N::Int, ::Val{WIDTH}) where {WIDTH}
    HWIDTH = WIDTH >> 1
    if j ≤ WIDTH
        return 1 + HWIDTH*(j - 1) + (j - 1)*j÷2
    elseif j ≤ N - WIDTH
        return (j - 1)*WIDTH + 1
    else
        jt = j - (N - WIDTH)
        return (N - WIDTH)*WIDTH + 1 +
               (WIDTH + HWIDTH + 1)*(jt - 1) - (jt - 1)*jt÷2
    end
end

"""
    _make_loop_expr_adjoint(DIM, N, WIDTH) -> Expr

Construct the full nested loop for the adjoint operator along dimension `DIM`.
Uses a running `ptr` into `A.coeffs_T`. Boundary regions span `WIDTH` outputs on
each side. Each fiber resets `ptr = ptr_start` computed via `_ptr_for_j`.
"""
function _make_loop_expr_adjoint(DIM, N, WIDTH)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1

    preamble = quote
        head_range = max(1, local_rng[1]):min($WIDTH, local_rng[end])
        body_range = max($WIDTH + 1, local_rng[1]):min(size(x, $DIM) - $WIDTH, local_rng[end])
        tail_range = max(size(x, $DIM) - $WIDTH + 1, local_rng[1]):min(size(x, $DIM), local_rng[end])
        has_head   = global_idx == 1 && local_rng[1] ≤ $WIDTH
        has_tail   = global_idx + local_rng[end] - 1 > size(A, 1) - $WIDTH
        _g_first_local = has_head ? local_rng[1] :
                         !isempty(body_range) ? first(body_range) :
                         first(tail_range)
        ptr_start  = _ptr_for_j(global_idx + _g_first_local - 1, size(A, 1), Val($WIDTH))
    end

    head_kernel = _make_kernel_variable(DIM, WIDTH, N, :($index + $HWIDTH), 1)
    body_kernel = _make_kernel_fixed(DIM, WIDTH, N, :coeffs_T, :($index - $HWIDTH))
    tail_kernel = _make_kernel_variable(DIM, WIDTH, N,
                      :(size(x, $DIM) - $index + $HWIDTH + 1), :($index - $HWIDTH))

    inner_head = head_kernel
    inner_body = body_kernel
    inner_tail = tail_kernel
    for d in 1:DIM-1
        rng        = :(1:$(Symbol(:N, d)))
        inner_head = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_head)
        inner_body = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_body)
        inner_tail = Expr(:for, Expr(:(=), Symbol(:i, d), rng), inner_tail)
    end

    head_advance = :(ptr += $index + $HWIDTH)
    body_advance = :(ptr += $WIDTH)
    tail_advance = :(ptr += size(x, $DIM) - $index + $HWIDTH + 1)

    dim_block = quote
        ptr = ptr_start
        if has_head
            for $index in head_range
                $inner_head
                $head_advance
            end
        end
        for $index in body_range
            $inner_body
            $body_advance
        end
        if has_tail
            for $index in tail_range
                $inner_tail
                $tail_advance
            end
        end
    end

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
writing the result into `y`. Non-distributed entry point.
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::DiffMatrix{T, WIDTH},
                            x::AbstractArray{S, N},
                             ::Val{DIM}  = Val(1)) where {T, S, N, WIDTH, DIM}
    size(x, DIM) == size(y, DIM) == size(A, 1) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), 1, 1:size(x, DIM))
end

"""
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}, global_idx, local_rng) -> y

Distribution-aware backend for the forward operator. `@generated` for the concrete
`(T, N, WIDTH, DIM)` combination.

- `global_idx`: global index of `local_rng[1]` in the row numbering of `A`.
- `local_rng`: local portion of dimension `DIM` to process.
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
`DIM`, writing the result into `y`. Non-distributed entry point.
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::AdjointDiffMatrix{T, WIDTH},
                            x::AbstractArray{S, N},
                             ::Val{DIM}  = Val(1)) where {T, S, N, WIDTH, DIM}
    size(x, DIM) == size(y, DIM) == size(A, 1) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), 1, 1:size(x, DIM))
end

"""
    LinearAlgebra.mul!(y, A::AdjointDiffMatrix, x, ::Val{DIM}, global_idx, local_rng) -> y

Distribution-aware backend for the adjoint operator. `@generated` for the concrete
`(T, N, WIDTH, DIM)` combination.

- `global_idx`: global index of `local_rng[1]`. Set to `1` for non-distributed calls.
- `local_rng`: local portion of dimension `DIM` to process.
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

Evaluate row `i` of the forward operator `A` applied to `x`. The sum is unrolled
at generation time.
"""
@generated function LinearAlgebra.mul!(A::DiffMatrix{T, WIDTH},
                                       x::AbstractVector,
                                       i::Int) where {T, WIDTH}
    quote
        N = length(x)

        size(A, 2) == N || throw(DimensionMismatch())

        left = clamp(i - $WIDTH >> 1, 1, N - $WIDTH + 1)
        ptr  = (i - 1) * $WIDTH + 1

        val = A.coeffs[ptr] * x[left]
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            val += A.coeffs[ptr + p] * x[left + p]
        end

        return val
    end
end
