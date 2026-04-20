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
    _make_kernel_expr(DIM, WIDTH, N, base) -> Expr

Build the expression for one stencil evaluation at a single grid point.

The generated code computes the finite-difference sum along dimension `DIM`
using a stencil of width `WIDTH`, with `base` giving the first index of the
local stencil window. The result is written to `y` at the current loop indices.

Conceptually, the emitted code evaluates

    y[...] = sum(A.coeffs[1+p, global_idx-1+iDIM] * x[..., base+p, ...] for p = 0:WIDTH-1)

where `iDIM = Symbol(:i, DIM)` is the loop variable along the differentiated
dimension. The summation is unrolled at code-generation time.

# Arguments
- `DIM`: Differentiation dimension.
- `WIDTH`: Number of stencil points.
- `N`: Total number of dimensions of `x` and `y`.
- `base`: Expression giving the first index of the stencil window along `DIM`.

# Returns
- An expression that computes the stencil value and stores it into `y`.

# Expected runtime bindings
The generated expression assumes the following names are available in scope:
- `A`: Differentiation operator.
- `x`: Input array.
- `y`: Output array.
- `global_idx`: Global row offset into `A.coeffs`.

# Typical values of `base`
- `:1` for the head region
- `:(iDIM - HWIDTH)` for the interior region
- `:(size(x, DIM) - WIDTH + 1)` for the tail region

# Notes
`global_idx` is `1` in the non-distributed case and larger when the operator is
applied to a local slice of a globally indexed dimension.
"""
function _make_kernel_expr(DIM, WIDTH, N, base)
    index = Symbol(:i, DIM)
    return quote
        s = A.coeffs[1, global_idx-1+$index] * $(_make_ref(:x, base, DIM, N))
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            s += A.coeffs[1+p, global_idx-1+$index] * $(_make_ref(:x, :(($base) + p), DIM, N))
        end
        $(_make_ref(:y, index, DIM, N)) = s
    end
end


"""
    _make_loop_expr(DIM, N, WIDTH) -> Expr

Construct the full nested loop expression that applies a finite-difference
stencil along dimension `DIM`.

The generated code splits the differentiated axis into three regions:

- **head**: points near the left boundary, where the stencil is pinned to the
  start of the domain
- **body**: interior points, where the stencil is centred
- **tail**: points near the right boundary, where the stencil is pinned to the
  end of the domain

Separate kernels are generated for these three cases so that the stencil
accumulation itself contains no runtime branching.

# Arguments
- `DIM`: Differentiation dimension.
- `N`: Number of array dimensions.
- `WIDTH`: Stencil width. This is expected to be odd.

# Returns
- An expression containing the runtime preamble and the complete loop nest.

# Structure of the generated code
1. A preamble computes `head_range`, `body_range`, `tail_range`, and the flags
   `has_head` and `has_tail`.
2. The stencil kernels for head, body, and tail are generated separately.
3. These kernels are wrapped in loops over all dimensions so that iteration
   follows Julia's column-major layout, with dimension 1 innermost.

# Expected runtime bindings
The emitted code expects the following variables to be defined:
- `A`, `x`, `y`
- `global_idx::Int`
- `local_rng::UnitRange`
- `N1, N2, ..., NN`, holding the sizes of `y`

# Distributed behaviour
Boundary kernels are only executed when `global_idx` and `local_rng` indicate
that the current process owns the corresponding boundary rows of `A`. Interior
subdomains therefore execute only the body kernel.

# Notes
This function only builds the loop AST; it does not execute anything itself.
"""
function _make_loop_expr(DIM, N, WIDTH)
    index = Symbol(:i, DIM)
    HWIDTH = WIDTH >> 1

    # ---- preamble: compute ranges and boundary flags at runtime ----
    # These are all loop-invariant; LLVM will hoist the has_head/has_tail
    # checks outside the generated loop automatically.
    preamble = quote
        # index ranges for each region along dimension DIM, clamped to local_rng
        head_range = max(1, local_rng[1]):min($HWIDTH, local_rng[end])
        body_range = max($HWIDTH + 1, local_rng[1]):min(size(x, $DIM) - $HWIDTH, local_rng[end])
        tail_range = max(size(x, $DIM) - $HWIDTH + 1, local_rng[1]):min(size(x, $DIM), local_rng[end])

        # this process owns the left boundary rows of A
        has_head = global_idx == 1 && local_rng[1] ≤ $HWIDTH
        # this process owns the right boundary rows of A
        has_tail = global_idx + local_rng[end] - 1 > size(A, 1) - $HWIDTH
    end

    # ---- kernels: one per region, base expression pinned at generation time ----
    head_kernel = _make_kernel_expr(DIM, WIDTH, N, :1)
    body_kernel = _make_kernel_expr(DIM, WIDTH, N, :($index - $HWIDTH))
    tail_kernel = _make_kernel_expr(DIM, WIDTH, N, :(size(x, $DIM) - $WIDTH + 1))

    # ---- phase 1: wrap kernels in inner loops for d < DIM ----
    # Iterating 1:DIM-1 means d=1 is wrapped first and ends up innermost,
    # keeping i1 as the fastest-running index (column-major order).
    inner_head = head_kernel
    inner_body = body_kernel
    inner_tail = tail_kernel
    for d in 1:DIM-1
        rng = :(1:$(Symbol(:N, d)))
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
        rng = :(1:$(Symbol(:N, d)))
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

This is the standard non-distributed entry point. It applies the operator over
the full extent of the differentiated dimension by forwarding to the lower-level
method with `global_idx = 1` and `local_rng = 1:size(x, DIM)`.

# Arguments
- `y`: Destination array.
- `A`: Finite-difference operator.
- `x`: Source array.
- `Val(DIM)`: Differentiation dimension, known at compile time. Defaults to `Val(1)`.

# Returns
- `y`

# Requirements
- `x` and `y` must have the same dimensionality.
- `size(x, DIM) == size(y, DIM) == size(A.coeffs, 2)`.

# Notes
Only the size of the differentiated dimension is checked here. The generated
implementation assumes the remaining dimensions of `x` and `y` are compatible.
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N}, 
                            A::DiffMatrix{T, WIDTH}, 
                            x::AbstractArray{S, N},
                             ::Val{DIM}=Val(1)) where {T, S, N, WIDTH, DIM}
    size(x, DIM) == size(y, DIM) == size(A.coeffs, 2) ||
        throw(ArgumentError("inconsistent inputs size"))
    return LinearAlgebra.mul!(y, A, x, Val(DIM), 1, 1:size(x, DIM))
end


"""
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}, global_idx, local_rng) -> y

Apply the finite-difference operator `A` to `x` along dimension `DIM`, writing
the result into `y`, over a specified local subrange of that dimension.

This method is the distribution-aware backend. It is implemented as a generated
function so that, for fixed `N`, `WIDTH`, and `DIM`, the loop nest, stencil
unrolling, and head/body/tail kernels are all specialised at compile time.

# Arguments
- `y`: Destination array.
- `A`: Finite-difference operator.
- `x`: Source array.
- `Val(DIM)`: Differentiation dimension, known at compile time.
- `global_idx::Int`: Global index of the first element in `local_rng`, measured
  in the row numbering used by `A.coeffs`.
- `local_rng::UnitRange`: Local portion of dimension `DIM` to process.

# Returns
- `y`

# Runtime checks
The method verifies that:
- `DIM` is a valid array dimension
- `size(x, DIM) == size(y, DIM)`
- `local_rng` lies within `1:size(x, DIM)`
- `global_idx + local_rng[end] - 1 ≤ size(A, 1)`

# Implementation details
- The generated code splits the differentiated axis into head, body, and tail
  regions.
- Boundary kernels are emitted only where needed; interior subdomains execute
  only the centred stencil.
- The sizes of `y` are unpacked into symbols `N1, N2, ...` so that the loop
  bounds can be embedded directly in the generated AST.
- The full loop nest is wrapped in `@inbounds` after the initial sanity checks.

# Notes
Use the higher-level `mul!(y, A, x, Val(DIM))` method unless you explicitly need
distributed or blockwise application.
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{T, N},
                                       A::DiffMatrix{TD, WIDTH},
                                       x::AbstractArray{T, N},
                                        ::Val{DIM},
                              global_idx::Int,
                               local_rng::UnitRange) where {T, TD, N, WIDTH, DIM}
    # safety checks
    DIM in 1:N || throw(ArgumentError("inconsistent differentiation dimension"))

    # build the full nested loop expression at compile time
    block = _make_loop_expr(DIM, N, WIDTH)

    # symbols for the sizes of each dimension, e.g. N1, N2, N3
    Ni = [Symbol(:N, d) for d in 1:N]

    return quote
        # runtime sanity checks
        size(x, $DIM) == size(y, $DIM) ||
            throw(ArgumentError("inconsistent inputs size"))
        local_rng[1] > 0 && local_rng[end] ≤ size(x, $DIM) ||
            throw(ArgumentError("out of bounds local range specification"))
        global_idx + local_rng[end] - 1 ≤ size(A, 1) ||
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
    LinearAlgebra.mul!(A::DiffMatrix{T,WIDTH}, x::AbstractVector, i::Int)

Evaluate row `i` of the finite-difference operator `A` on the vector `x`.

This computes a single stencil application, returning the scalar

    sum(A.coeffs[1+p, i] * x[left+p] for p = 0:WIDTH-1)

where `left` is the clamped starting index of the stencil window centered as
closely as possible around `i`.

# Arguments
- `A`: Finite-difference operator.
- `x`: Input vector.
- `i`: Target row / evaluation index.

# Returns
- The scalar value of the operator applied at index `i`.

# Behaviour near boundaries
The stencil window is shifted as needed so that it remains within the valid
index range of `x`.

# Requirements
- `length(x) == size(A, 2)`.

# Notes
The stencil sum is unrolled at generation time.
"""
@generated function LinearAlgebra.mul!(A::DiffMatrix{T, WIDTH}, x::AbstractVector, i::Int) where {T, WIDTH}
    quote
        # size of vector
        N = length(x)

        # check size
        size(A, 2) == N || throw(DimensionMismatch())

        # index of the first element of the stencil
        left = clamp(i - $WIDTH >> 1, 1, N - $WIDTH + 1)

        # expand expressions
        val = A.coeffs[1, i] * x[left]
        Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
            val += A.coeffs[1+p, i] * x[left+p]
        end

        return val
    end
end