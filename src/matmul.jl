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
function _make_kernel(DIM, base, WIDTH, N, OFFSET)
    return quote
        s = A.coeffs[1, $OFFSET + $(__VARS__[DIM])] * $(_make_ref(:x, base, DIM, N))
        Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
            s += A.coeffs[1 + p, $OFFSET + $(__VARS__[DIM])] * $(_make_ref(:x, :( ($base) + p), DIM, N))
        end
        $(_make_ref(:y, __VARS__[DIM], DIM, N)) = s
    end
end

"""
    _make_loop_expr(rows, DIM, N, block) -> Expr

Construct an `Expr` representing a nested `for` loop of depth `N`.

The generated expression has the structure

    for i₁ = r₁
        for i₂ = r₂
            ⋮
                for iₙ = rₙ
                    block
                end
            ⋮
        end
    end

where the iteration ranges `r_d` are defined as follows:

- For `d == DIM`, the loop range is `rows`.
- For all other dimensions `d ≠ DIM`, the loop range is `1:N_d`,
  where `N_d` is represented symbolically as `Symbol(:N, d)`.

The loop indices are taken from `__VARS__`, and the loops are generated
in descending order (`N:-1:1`) to maximise cache-locality in line with
column-major ordering of array data. The function supports loop depths
up to and including `N = 4`.

# Arguments
- `rows`: An expression specifying the iteration range for dimension `DIM`.
- `DIM`: The dimension index whose loop iterates over `rows`.
- `N`: The total number of nested loops to generate.
- `block`: The expression forming the body of the innermost loop.

# Returns
- `Expr`: A Julia expression representing the constructed nested loop.
"""
function _make_loop_expr(rows, DIM, N, block)
    # create loop head
    Nd_loop_expr = [Expr(:(=), __VARS__[d], d==DIM ? rows : Expr(:call, :(:), 1, Symbol(:N, d))) for d in N:-1:1]

    # define and return full loop expressions
    return Expr(:for, Expr(:block, Nd_loop_expr...), block)
end

"""
    LinearAlgebra.mul!(y, A::DiffMatrix, x, ::Val{DIM}=Val(1)) -> y

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
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::DiffMatrix{T, WIDTH},
                            x::AbstractArray{S, N},
                             ::Val{DIM}=Val(1)) where {T, S, N, WIDTH, DIM}
    # check sizes match along the dimension we differentiate along
    size(x, DIM) == size(y, DIM) == size(A.coeffs, 2) || throw(ArgumentError("inconsistent inputs size"))

    # perform multiplication
    LinearAlgebra.mul!(y, A, x, Val(:hb), Val(0), Val(                        1:(WIDTH>>1)               ), Val(size(x, DIM)), Val(DIM))
    LinearAlgebra.mul!(y, A, x, Val(:b ), Val(0), Val(           ((WIDTH>>1)+1):(size(x, DIM)-(WIDTH>>1))), Val(size(x, DIM)), Val(DIM))
    LinearAlgebra.mul!(y, A, x, Val(:bt), Val(0), Val((size(x, DIM)-(WIDTH>>1)):size(x, DIM)             ), Val(size(x, DIM)), Val(DIM))

    return y
end

@generated function LinearAlgebra.mul!(y::AbstractArray{T, N},
                                       A::DiffMatrix{TD, WIDTH},
                                       x::AbstractArray{T, N},
                                        ::Val{CASE},
                                        ::Val{OFFSET},
                                        ::Val{RNG},
                                        ::Val{LENGTH},
                                        ::Val{DIM}=Val(1)) where {T, TD, N, WIDTH, CASE, OFFSET, RNG, LENGTH, DIM}
    # safety checks
    N   in 1:4                         || throw(ArgumentError("invalid array dimension"))
    DIM in 1:N                         || throw(ArgumentError("inconsistent differentiation dimension"))
    (RNG[1] > 0 && RNG[end] <= LENGTH) || throw(ArgumentError("invalid range"))

    # get correct expression for given region
    block =
        if CASE == :hb
            head_kernel = _my_head_mul!(WIDTH, DIM, N)
            body_kernel = _my_body_mul!(WIDTH, 0, (WIDTH >> 1) + 1:RNG[end], DIM, N)
            quote
                $head_kernel; $body_kernel
            end
        elseif CASE == :b
            body_kernel = _my_body_mul!(WIDTH, OFFSET, RNG, DIM, N)
            quote
                $body_kernel
            end
        elseif CASE == :bt
            body_kernel = _my_body_mul!(WIDTH, OFFSET, RNG[1]:(LENGTH - (WIDTH >> 1)), DIM, N)
            tail_kernel = _my_tail_mul!(WIDTH, LENGTH, OFFSET, DIM, N)
            quote
                $body_kernel; $tail_kernel
            end
        else
            throw(ArgumentError("invalid case argument"))
        end

    # define N1, N2, N3, N4
    Ni = [Symbol(:N, d) for d in 1:N]

    output = quote
        # size of array as a tuple, e.g.
        # N1, N2, N3 = size(y)
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))

        # main differentiation loop
        @inbounds begin
            $block
        end

        return y
    end

    return output
end

_my_head_mul!(WIDTH, DIM, N) = _make_loop_expr(:(1:$(WIDTH >> 1)), DIM, N, _make_kernel(DIM, :(1), WIDTH, N, 0))
_my_body_mul!(WIDTH, OFFSET, RNG, DIM, N) = _make_loop_expr(:($RNG), DIM, N, _make_kernel(DIM, :($(__VARS__[DIM]) - $(WIDTH >> 1)), WIDTH, N, OFFSET))
_my_tail_mul!(WIDTH, LENGTH, OFFSET, DIM, N) = _make_loop_expr(:($(LENGTH - (WIDTH >> 1) + 1):$LENGTH), DIM, N, _make_kernel(DIM, :($(LENGTH - WIDTH + 1)), WIDTH, N, OFFSET))


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
