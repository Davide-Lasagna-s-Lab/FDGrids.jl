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


"""
    _local_range(global_rng, global_idx, local_rng) -> UnitRange

Translate the intersection of a global row range with `local_rng` into local
indices. `global_idx` is the global row represented by local index `1`.
"""
@inline function _local_range(global_rng::UnitRange,
                              global_idx::Int,
                              local_rng::UnitRange)
    first_local = first(global_rng) - global_idx + 1
    last_local  = last(global_rng)  - global_idx + 1
    return max(first(local_rng), first_local):min(last(local_rng), last_local)
end


# ================================================================================
# KERNEL BUILDERS
# ================================================================================

"""
    _make_kernel_fixed(DIM, WIDTH, N, base_expr, ADD) -> Expr

Emit a stencil kernel that reads exactly WIDTH coefficients from
`A.coeffs[ptr..]` and WIDTH x-values starting at `x[base_expr..]`.
The dot-product is fully unrolled at code-generation time.

Covers all forward regions and the adjoint body:
  - forward head:  base_expr=1
  - forward/adjoint body: base_expr=:(index - HWIDTH)
  - forward tail:  base_expr=:(size(x,DIM) - WIDTH + 1)
"""
function _make_kernel_fixed(DIM, WIDTH, N, base_expr, ADD)
    index = Symbol(:i, DIM)
    yref = _make_ref(:y, index, DIM, N)
    store = ADD ? :($yref += s) : :($yref = s)

    return quote
        s = A.coeffs[ptr] * $(_make_ref(:x, base_expr, DIM, N))
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            s += A.coeffs[ptr + p] * $(_make_ref(:x, :(($base_expr) + p), DIM, N))
        end
        $store
    end
end

"""
    _make_kernel_variable(DIM, WIDTH, N, nrows_expr, base_expr, ADD) -> Expr

Emit a stencil kernel that reads a variable number of terms from
`A.coeffs[ptr..]` and `x[base_expr + _k - 1]` for `_k in 1:nrows_j`.

Used for adjoint boundary outputs whose stencil row-count varies:
  - adjoint head: nrows_expr=:(index + HWIDTH),                   base_expr=1
  - adjoint tail: nrows_expr=:(size(x,DIM) - index + HWIDTH + 1), base_expr=:(index - HWIDTH)
"""
function _make_kernel_variable(DIM, WIDTH, N, nrows_expr, base_expr, ADD)
    index = Symbol(:i, DIM)
    yref = _make_ref(:y, index, DIM, N)
    store = ADD ? :($yref += s) : :($yref = s)

    return quote
        nrows_j = $nrows_expr
        s = zero(eltype(y))
        for _k in 1:nrows_j
            s += A.coeffs[ptr + _k - 1] * $(_make_ref(:x, :(($base_expr) + _k - 1), DIM, N))
        end
        $store
    end
end


# ================================================================================
# LOOP BUILDERS — forward and adjoint
# ================================================================================

"""
    _make_loop_expr_forward(DIM, N, WIDTH, ADD) -> Expr

Construct the full nested loop for the forward operator along dimension `DIM`.
Uses a running `ptr` into `A.coeffs` that advances by `WIDTH` per output.
Each fiber (outer-loop iteration) resets `ptr = ptr_start`.
"""
function _make_loop_expr_forward(DIM, N, WIDTH, ADD)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1

    preamble = quote
        # DiffMatrix has three global regions: a left-boundary head, a centered
        # body, and a right-boundary tail. Translate their intersections with
        # this slab back to local indices. This keeps a middle slab in the body
        # even when a halo-aware array exposes ghost cells outside its axes.
        head_range = _local_range(1:$HWIDTH,
                                  global_idx, local_rng)
        body_range = _local_range(($HWIDTH + 1):(size(A, 1) - $HWIDTH),
                                  global_idx, local_rng)
        tail_range = _local_range((size(A, 1) - $HWIDTH + 1):size(A, 1),
                                  global_idx, local_rng)
        has_head   = !isempty(head_range)
        has_tail   = !isempty(tail_range)

        # The coefficient pointer is global-row based. Find the first local row
        # that this call will actually compute and jump directly to its row in
        # the compact row-major coefficient vector.
        _f_local   = has_head ? local_rng[1] :
                     !isempty(body_range) ? first(body_range) :
                     first(tail_range)
        ptr_start  = (global_idx + _f_local - 2) * $WIDTH + 1
    end

    head_kernel = _make_kernel_fixed(DIM, WIDTH, N, 1, ADD)
    body_kernel = _make_kernel_fixed(DIM, WIDTH, N, :($index - $HWIDTH), ADD)
    tail_kernel = _make_kernel_fixed(DIM, WIDTH, N, :(size(x, $DIM) - $WIDTH + 1), ADD)

    inner_head = head_kernel
    inner_body = body_kernel
    inner_tail = tail_kernel
    for d in 1:DIM-1
        rng        = :(1:$(Symbol(:N, d)))
        # Dimensions before DIM must be nested inside the differentiated loop
        # so that all non-DIM coordinates are visited for each stencil row.
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


"""
    _make_loop_expr_adjoint(DIM, N, WIDTH, ADD) -> Expr

Construct the full nested loop for the adjoint operator along dimension `DIM`.
Uses a running `ptr` into `A.coeffs`. Boundary regions span `WIDTH` outputs on
each side. Each fiber resets `ptr = ptr_start` computed via `_ptr_for_j`.
"""
function _make_loop_expr_adjoint(DIM, N, WIDTH, ADD)
    index  = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1

    preamble = quote
        # Adjoint output rows use variable-length global head/tail regions.
        # Translate their intersections with this slab back to local indices.
        # The body has fixed WIDTH coefficients, but the first/last WIDTH rows
        # do not.
        head_range = _local_range(1:$WIDTH,
                                  global_idx, local_rng)
        body_range = _local_range(($WIDTH + 1):(size(A, 1) - $WIDTH),
                                  global_idx, local_rng)
        tail_range = _local_range((size(A, 1) - $WIDTH + 1):size(A, 1),
                                  global_idx, local_rng)
        has_head   = !isempty(head_range)
        has_tail   = !isempty(tail_range)

        # Unlike the forward operator, the pointer cannot be computed by a
        # simple WIDTH stride in the head/tail. _ptr_for_j encodes the compact
        # variable-length adjoint storage layout.
        _g_first_local = has_head ? local_rng[1] :
                         !isempty(body_range) ? first(body_range) :
                         first(tail_range)
        ptr_start  = _ptr_for_j(global_idx + _g_first_local - 1, size(A, 1), Val($WIDTH))
    end

    head_kernel = _make_kernel_variable(DIM, WIDTH, N, :($index + $HWIDTH), 1, ADD)
    body_kernel = _make_kernel_fixed(DIM, WIDTH, N, :($index - $HWIDTH), ADD)
    tail_kernel = _make_kernel_variable(DIM, WIDTH, N,
                      :(size(x, $DIM) - $index + $HWIDTH + 1), :($index - $HWIDTH), ADD)

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
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}=Val(1), ::Val{ADD}=Val(false)) -> y

Apply the forward finite-difference operator `A` to `x` along dimension `DIM`,
writing the result into `y`. Non-distributed entry point. If `ADD` is `true`,
add the differentiated values to the existing contents of `y`.

For vectors, the default `DIM=1` applies the usual matrix-vector action. For
higher-dimensional arrays, each one-dimensional fiber along `DIM` is
differentiated independently.

# Examples
```julia
using LinearAlgebra

xs = range(-1, 1; length = 32)
D  = DiffMatrix(xs, 5, 1)

u  = sin.(xs)
du = similar(u)
mul!(du, D, u)

A  = repeat(reshape(u, :, 1), 1, 4)
dA = similar(A)
mul!(dA, D, A, Val(1))
```
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::DiffMatrix{T, WIDTH},
                            x::AbstractArray{S, N},
                             ::Val{DIM}  = Val(1),
                             ::Val{ADD}  = Val(false)) where {T, S, N, WIDTH, DIM, ADD}
    size(x, DIM) == size(y, DIM) == size(A, 1) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), 1, 1:size(x, DIM), Val(ADD))
end

"""
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}, global_idx, local_rng, ::Val{ADD}=Val(false)) -> y

Apply the forward operator to selected rows of domain-decomposed storage.

- `global_idx`: global row index corresponding to local index `1` of `x`/`y`.
- `local_rng`: local portion of dimension `DIM` to process.
- `ADD`: when `true`, add the result into `y` instead of overwriting it.

This method is intended for domain-decomposed callers. Ordinary local arrays
should use `mul!(y, A, x, Val(DIM))`.

`x` must provide every stencil entry needed to evaluate `local_rng`. It may
store halo rows inside its ordinary axes and shift `local_rng` inward, or expose
ghost cells through halo-aware scalar indices outside those axes. In both
cases, `global_idx` describes local index `1`, not `first(local_rng)`.
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{T, N},
                                       A::DiffMatrix{TD, WIDTH},
                                       x::AbstractArray{T, N},
                                        ::Val{DIM},
                              global_idx::Int,
                               local_rng::UnitRange,
                                        ::Val{ADD}=Val(false)) where {T, TD, N, WIDTH, DIM, ADD}
    DIM in 1:N || throw(ArgumentError("inconsistent differentiation dimension"))
    ADD isa Bool || throw(ArgumentError("ADD must be true or false"))

    block = _make_loop_expr_forward(DIM, N, WIDTH, ADD)
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
    LinearAlgebra.mul!(y, A::AdjointDiffMatrix, x, ::Val{DIM}=Val(1), ::Val{ADD}=Val(false)) -> y

Apply the transposed finite-difference operator `A = D*` to `x` along dimension
`DIM`, writing the result into `y`. Non-distributed entry point. If `ADD` is
`true`, add the differentiated values to the existing contents of `y`.

# Examples
```julia
using LinearAlgebra

xs = range(-1, 1; length = 32)
D  = DiffMatrix(xs, 5, 1)
Dt = adjoint(D)

y = similar(collect(xs))
mul!(y, Dt, cos.(xs))
```
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::AdjointDiffMatrix{T, WIDTH},
                            x::AbstractArray{S, N},
                             ::Val{DIM}  = Val(1),
                             ::Val{ADD}  = Val(false)) where {T, S, N, WIDTH, DIM, ADD}
    size(x, DIM) == size(y, DIM) == size(A, 1) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), 1, 1:size(x, DIM), Val(ADD))
end

"""
    LinearAlgebra.mul!(y, A::AdjointDiffMatrix, x, ::Val{DIM}, global_idx, local_rng, ::Val{ADD}=Val(false)) -> y

Apply the adjoint operator to selected rows of domain-decomposed storage.

- `global_idx`: global row index corresponding to local index `1` of `x`/`y`.
- `local_rng`: local portion of dimension `DIM` to process.
- `ADD`: when `true`, add the result into `y` instead of overwriting it.

This method is intended for distributed or slab-local application of an adjoint
operator. Ordinary local arrays should use `mul!(y, A, x, Val(DIM))`.

`x` must provide every stencil entry needed to evaluate `local_rng`. It may
store halo rows inside its ordinary axes and shift `local_rng` inward, or expose
ghost cells through halo-aware scalar indices outside those axes. In both
cases, `global_idx` describes local index `1`, not `first(local_rng)`.
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{T, N},
                                       A::AdjointDiffMatrix{TD, WIDTH},
                                       x::AbstractArray{T, N},
                                        ::Val{DIM},
                              global_idx::Int,
                               local_rng::UnitRange,
                                        ::Val{ADD}=Val(false)) where {T, TD, N, WIDTH, DIM, ADD}
    DIM in 1:N || throw(ArgumentError("inconsistent differentiation dimension"))
    ADD isa Bool || throw(ArgumentError("ADD must be true or false"))

    block = _make_loop_expr_adjoint(DIM, N, WIDTH, ADD)
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

This is useful when only one differentiated value is needed. It does not mutate
`A` or `x`.

# Examples
```julia
using LinearAlgebra

xs = range(-1, 1; length = 16)
D  = DiffMatrix(xs, 5, 1)
du_at_3 = mul!(D, sin.(xs), 3)
```
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
