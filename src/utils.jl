"""
    get_weights(ξ, x, m) -> Matrix

Compute finite-difference interpolation and differentiation weights at `ξ`
using the nodes in `x`.

The returned matrix has size `(length(x), m+1)`. Column `k+1` contains weights
for the `k`-th derivative, so

```julia
dot(get_weights(ξ, x, m)[:, k+1], f.(x))
```

approximates `f^(k)(ξ)`. Column `1` is therefore the interpolation rule.

The nodes in `x` must be distinct. They may be non-uniform and need not be
centered around `ξ`.

The implementation is Fornberg's recursive algorithm for finite-difference
weights on arbitrarily spaced nodes, adapted from the `nscouette` code
(Marc Avila, ZARM, Bremen University) and Appendix C of Bengt Fornberg's
*A Practical Guide to Pseudospectral Methods*.

# Examples
```julia
x = [-1.0, 0.0, 1.0]
c = get_weights(0.0, x, 2)
c[:, 2]  # first-derivative weights
```
"""
function get_weights(ξ::Real,
                     x::AbstractVector{<:Real},
                     m::Int)

    n = length(x) - 1
    coeffs = zeros(n+1, m+1)

    # This is spaghetti code written in the 1990's :( but
    # it works and I thank it exists! Note Julia is 1 based
    c1 = 1.0
    c4 = x[1] - ξ
    coeffs[1, 1] = 1.0
    for i = 1:n
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i+1] - ξ
        for j = 0:i-1
            c3 = x[i+1] - x[j+1]
            c2 = c2*c3
            if j == i-1
                for k = mn:-1:1
                   coeffs[i+1, k+1] = c1*(k*coeffs[i, k] - c5*coeffs[i, k+1])/c2
                end
                coeffs[i+1, 1] = -c1*c5*coeffs[i, 1]/c2
            end
            for k = mn:-1:1
                coeffs[j+1, k+1] = (c4*coeffs[j+1, k+1] - k*coeffs[j+1, k])/c3
            end
            coeffs[j+1, 1] = c4*coeffs[j+1, 1]/c3
        end
        c1 = c2
    end
    return coeffs
end

"""
    get_coeffs(xs, width, order) -> Matrix

Compute the compact row-wise coefficients for a `DiffMatrix`.

`xs` are the grid points, `width` is the odd stencil width, and `order` is the
derivative order. The result has size `(width, length(xs))`; column `i` contains
the finite-difference weights for the derivative at `xs[i]`.

Interior points use centered stencils. Boundary points use the nearest valid
one-sided stencil of the same width, so the storage layout is uniform across all
rows of the eventual `DiffMatrix`.

# Examples
```julia
xs = range(-1, 1; length = 8)
C  = get_coeffs(xs, 5, 1)
```
"""
function get_coeffs(xs::AbstractVector{<:Real}, width::Int, order::Int)

    # make sure we have an odd number of points
    width % 2 == 1 || throw(ArgumentError("width must be odd"))

    # either first or second derivative
    width > order || throw(ArgumentError("stencil width must be 
        larger than derivative order, got width=$width, order=$order"))

    # this is what these matrices look like for a stencil of width 5 on 9 grid points
    # x 2 3 4 5
    # 1 x 3 4 5
    # 1 2 x 4 5
    #   2 3 x 5 6
    #     3 4 x 6 7
    #       4 5 x 7 8
    #         5 6 x 8 9
    #         5 6 7 x 9
    #         5 6 7 8 x

    # number of grid points
    N = length(xs)

    # init coefficient matrices, one column with stencil coefficients per grid point
    coeffs = zeros(width, N)

    # for every grid point
    for i = 1:N
        # define the span of the stencil (see figure 4.7-2 in Fornberg's book).
        # this makes sure that the width of the stencil is not smaller than prescribed
        left  = clamp(i - width>>1,     1, N - width + 1)
        right = clamp(i + width>>1, width, N)

        # get coefficients
        weights = get_weights(xs[i], view(xs, left:right), order+2)

        # store line - note that the first column of c is the zero
        # order derivative (i.e. interpolation)
        coeffs[:, i] .= weights[:, order+1]
    end

    return coeffs
end

"""
    basis_vector(i, P, [T=Float64]) -> Vector{T}

Return the `i`-th canonical basis vector of length `P` with element type `T`.
"""
basis_vector(i, P, ::Type{T}=Float64) where {T} = (out = zeros(T, P); out[i] = 1; out)
