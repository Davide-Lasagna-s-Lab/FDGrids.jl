# complete set of loop variables and array dimensions (up to array of dimension 4)
const __VARS__ = (:i, :j, :k, :l)
const __NS__ = (:N1, :N2, :N3, :N4)

"""
    _make_ref(array, expr, DIM, N) -> Expr

Construct an indexing AST (`Expr(:ref, ...)`) for an `N`-dimensional array symbol,
using the loop-variable symbols stored in `__VARS__`.

The returned expression has the form

- `array[i, j, k, ...]` in `N` dimensions,

except that the index in position `DIM` is replaced by the expression `expr`.

This is intended for code generation inside `@generated` methods.

# Arguments
- `array`: Symbol or expression representing the array being indexed (typically `:x` or `:y`).
- `expr`: Expression to be placed in slot `DIM` (e.g. `1`, `:base`, `:(base + p)`).
- `DIM`: Integer axis (1-based) whose index is replaced by `expr`.
- `N`: Total number of array dimensions.

# Requirements / Assumptions
- `1 ≤ DIM ≤ N`.

# Examples
- `_make_ref(:x, 1, 2, 4)` generates `x[i, 1, k, l]`.
- `_make_ref(:x, :(base + p), 3, 4)` generates `x[i, j, base + p, l]`.
"""
function _make_ref(array, expr, DIM, N)
    inds = ntuple(d -> d == DIM ? expr : __VARS__[d], N)
    return Expr(:ref, array, inds...)
end

"""
    _make_kernel(DIM, base, WIDTH, N) -> Expr

Build an AST (`quote ... end`) for applying a width-`WIDTH` finite-difference stencil
along axis `DIM` at the current grid location defined by the loop variables in `__VARS__`.

The generated code computes

- `s = Σ_{p=0}^{WIDTH-1} A.coeffs[1+p, q] * x[ ... , base + p, ... ]`

where `q = __VARS__[DIM]` is the loop variable for the differentiated axis, and the
stencil varies only in slot `DIM`. The result is written to the corresponding element
of `y` at the current indices.

The summation is unrolled at generation time using `Base.Cartesian.@nexprs`.

# Arguments
- `DIM`: Integer axis (1-based) the stencil is applied along.
- `base`: Expression for the first index of the stencil window placed in slot `DIM`
  (e.g. `1`, `:(q - hW)`, `:(Nq - WIDTH + 1)`).
- `WIDTH`: Stencil width (compile-time constant for unrolling).
- `N`: Total number of array dimensions.

# Requirements / Assumptions
- `A`, `x`, `y`, and `__VARS__` are in scope in the final emitted code.

# Notes
- `base` is treated as an AST; the term `(base) + p` is emitted as code,
  and `p` is substituted by `@nexprs` during macro expansion/unrolling.
- The assignment target is generated via `_make_ref` so the code works for any `N`.

# Examples
Typical usage is to build three kernels for head/body/tail regions, with different
`base` expressions, and splice them into the appropriate loop nests.
"""
function _make_kernel(DIM, base, WIDTH, N)
    return quote
        s = A.coeffs[1, $(__VARS__[DIM])] * $(_make_ref(:x, base, DIM, N))
        Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
            s += A.coeffs[1 + p, $(__VARS__[DIM])] * $(_make_ref(:x, :( ($base) + p), DIM, N))
        end
        $(_make_ref(:y, __VARS__[DIM], DIM, N)) = s
    end
end

"""
    LinearAlgebra.mul!(y, A, x, ::Val{DIM}=Val(1))

Apply a finite-difference differentiation operator stored in `A::DiffMatrix` to an
`N`-dimensional array `x` and write the result into `y`, differentiating along
dimension `DIM`.

This method is implemented as a `@generated` function and supports `N ∈ 1:4`.
`DIM` is a compile-time constant (`Val{DIM}`), so the generated code:

- specializes the loop nest for the chosen `(N, DIM)` pair,
- keeps the first array index (`i`) as the fastest-running loop index wherever possible,
- splits the differentiated axis into three regions (head/body/tail) to handle
  boundary stencils without branching in the inner loop,
- unrolls the stencil accumulation using `Base.Cartesian.@nexprs` with width `WIDTH`.

# Arguments
- `y::AbstractArray{S,N}`: output array (updated in-place).
- `A::DiffMatrix{T,WIDTH}`: differentiation operator. `A.coeffs` is assumed to be
  `WIDTH × size(x,DIM)`, with column `q` containing the stencil weights for output
  location `q` along the differentiated axis.
- `x::AbstractArray{S,N}`: input array.
- `::Val{DIM}`: compile-time differentiation dimension (default `Val(1)`).

# Type Parameters
- `S`: element type of `x` and `y`.
- `T`: coefficient type stored in `A`.
- `N`: array dimensionality (restricted to `1:4`).
- `WIDTH`: stencil width (compile-time constant for unrolling).
- `DIM`: differentiation dimension, `1 ≤ DIM ≤ N`.

# Behaviour and Assumptions
- Checks that `size(x, DIM) == size(y, DIM) == size(A.coeffs, 2)`.
- Assumes loop index symbols `__VARS__` (e.g. `(:i,:j,:k,:l)`) and size symbols
  `__NS__` (e.g. `(:N1,:N2,:N3,:N4)`) are available in the generator scope.
- Uses three stencil starting positions:
  - head: `base = 1`
  - body: `base = q - hWIDTH`
  - tail: `base = Nq - WIDTH + 1`
  where `q` is the loop variable associated with the differentiated dimension and
  `hWIDTH = WIDTH ÷ 2`.

# Implementation Notes
- The emitted code destructures `size(y)` into `N1, N2, ...` using a generated tuple
  assignment, e.g. `(N1,N2,N3) = size(y)`.
- Loop structure depends on `(N,DIM)` to avoid restarting the outer loops between
  head/body/tail segments for `DIM < N`, improving memory locality.

# Examples
Differentiate a 3D field `x` along the second dimension:

```julia
mul!(y, A, x, Val(2))
```
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       A::DiffMatrix{T, WIDTH},
                                       x::AbstractArray{S, N}, 
                                        ::Val{DIM} = Val(1), 
                                        ::Val{CASE} = Val((:h, :b, :t))) where {T, S, N, WIDTH, DIM, CASE}
    # sanity checks
    N in 1:4 ||
        throw(ArgumentError("inconsistent array dimension"))

    DIM in 1:N ||
        throw(ArgumentError("inconsistent differentiation dimension"))
    
    # array size of the dimension we differentiate along
    q = __VARS__[DIM]
    Nq = __NS__[DIM]

    # stencil half width
    hWIDTH = WIDTH >> 1 

    # kernel expressions, starting at different locations
    head_kernel = _make_kernel(DIM, :(1),                WIDTH, N)
    body_kernel = _make_kernel(DIM, :($q - $hWIDTH),     WIDTH, N)
    tail_kernel = _make_kernel(DIM, :($Nq - $WIDTH + 1), WIDTH, N)

    # ranges for the head, body and tail regions
    head_range = :(1:$hWIDTH)
    body_range = :((1 + $hWIDTH):($Nq - $hWIDTH))
    tail_range = :(($Nq - $hWIDTH + 1):$Nq)

    # main for loop block
    block =
        if N == 1
            head_part = (:h in CASE) ? :(for i = $head_range; $head_kernel; end) : :()
            body_part = (:b in CASE) ? :(for i = $body_range; $body_kernel; end) : :()
            tail_part = (:t in CASE) ? :(for i = $tail_range; $tail_kernel; end) : :()
            quote
                $head_part; $body_part; $tail_part
            end
        elseif N == 2
            if DIM == 1
                head_part = (:h in CASE) ? :(for i = $head_range; $head_kernel; end) : :()
                body_part = (:b in CASE) ? :(for i = $body_range; $body_kernel; end) : :()
                tail_part = (:t in CASE) ? :(for i = $tail_range; $tail_kernel; end) : :()
                quote
                    for j = 1:N2
                        $head_part; $body_part; $tail_part
                    end
                end
            else # DIM == 2
                head_part = (:h in CASE) ? :(for j = $head_range; for i = 1:N1; $head_kernel; end; end) : :()
                body_part = (:b in CASE) ? :(for j = $body_range; for i = 1:N1; $body_kernel; end; end) : :()
                tail_part = (:t in CASE) ? :(for j = $tail_range; for i = 1:N1; $tail_kernel; end; end) : :()
                quote
                    $head_part; $body_part; $tail_part
                end
            end
        elseif N == 3
            if DIM == 1
                head_part = (:h in CASE) ? :(for i = $head_range; $head_kernel; end) : :()
                body_part = (:b in CASE) ? :(for i = $body_range; $body_kernel; end) : :()
                tail_part = (:t in CASE) ? :(for i = $tail_range; $tail_kernel; end) : :()
                quote
                    for k = 1:N3
                        for j = 1:N2
                            $head_part; $body_part; $tail_part
                        end
                    end
                end
            elseif DIM == 2
                head_part = (:h in CASE) ? :(for j = $head_range; for i = 1:N1; $head_kernel; end; end) : :()
                body_part = (:b in CASE) ? :(for j = $body_range; for i = 1:N1; $body_kernel; end; end) : :()
                tail_part = (:t in CASE) ? :(for j = $tail_range; for i = 1:N1; $tail_kernel; end; end) : :()
                quote
                    for k = 1:N3
                        $head_part; $body_part; $tail_part
                    end
                end
            else # DIM == 3
                head_part = (:h in CASE) ? :(for k = $head_range, j = 1:N2, i = 1:N1; $head_kernel; end) : :()
                body_part = (:b in CASE) ? :(for k = $body_range, j = 1:N2, i = 1:N1; $body_kernel; end) : :()
                tail_part = (:t in CASE) ? :(for k = $tail_range, j = 1:N2, i = 1:N1; $tail_kernel; end) : :()
                quote
                    $head_part; $body_part; $tail_part
                end
            end
        else # N == 4
            if DIM == 1
                head_part = (:h in CASE) ? :(for i = $head_range; $head_kernel; end) : :()
                body_part = (:b in CASE) ? :(for i = $body_range; $body_kernel; end) : :()
                tail_part = (:t in CASE) ? :(for i = $tail_range; $tail_kernel; end) : :()
                quote
                    for l = 1:N4, k = 1:N3, j = 1:N2
                        $head_part; $body_part; $tail_part
                    end
                end
            elseif DIM == 2
                head_part = (:h in CASE) ? :(for j = $head_range, i = 1:N1; $head_kernel; end) : :()
                body_part = (:b in CASE) ? :(for j = $body_range, i = 1:N1; $body_kernel; end) : :()
                tail_part = (:t in CASE) ? :(for j = $tail_range, i = 1:N1; $tail_kernel; end) : :()
                quote
                    for l = 1:N4, k = 1:N3
                        $head_part; $body_part; $tail_part
                    end
                end
            elseif DIM == 3
                head_part = (:h in CASE) ? :(for k = $head_range, j = 1:N2, i = 1:N1; $head_kernel; end) : :()
                body_part = (:b in CASE) ? :(for k = $body_range, j = 1:N2, i = 1:N1; $body_kernel; end) : :()
                tail_part = (:t in CASE) ? :(for k = $tail_range, j = 1:N2, i = 1:N1; $tail_kernel; end) : :()
                quote
                    for l = 1:N4
                        $head_part; $body_part; $tail_part
                    end
                end
            else # DIM == 4
                head_part = (:h in CASE) ? :(for l = $head_range, k = 1:N3, j = 1:N2, i = 1:N1; $head_kernel; end) : :()
                body_part = (:b in CASE) ? :(for l = $body_range, k = 1:N3, j = 1:N2, i = 1:N1; $body_kernel; end) : :()
                tail_part = (:t in CASE) ? :(for l = $tail_range, k = 1:N3, j = 1:N2, i = 1:N1; $tail_kernel; end) : :()
                quote
                    $head_part; $body_part; $tail_part
                end
            end
        end
        
    # define N1, N2, ...
    Ni = [Symbol(:N, d) for d in 1:N]
    
    return quote
        # check sizes match along the dimension we differentiate along
        size(x, $DIM) == size(y, $DIM) == size(A.coeffs, 2) ||
            throw(ArgumentError("inconsistent inputs size"))

        # size of array as a tuple, e.g.
        # N1, N2, N3 = size(y)
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))

        @inbounds begin
            $block
        end
        
        return y
    end
end

# Find derivative of `x` at point `i` 
@generated function LinearAlgebra.mul!(A::DiffMatrix{T, WIDTH}, x::AbstractVector, i::Int) where {T, WIDTH}
    quote
        # size of vector
        N = length(x)

        # check size
        size(A, 2) == N || throw(DimensionMismatch())

        # index of the first element of the stencil
        left = clamp(i - $WIDTH>>1, 1, N - $WIDTH + 1)

        # expand expressions
        val = A.coeffs[1, i]*x[left]
        Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
            val += A.coeffs[1 + p, i]*x[left + p]
        end
        
        return val
    end
end