export quadweights, _quadweights

# The primary API for grid points and quadrature weights is `grid(M, l, h, dist)`
# in grids.jl, which returns both together. The functions below are retained for
# backward compatibility with existing code that calls `quadweights(xs, order)`.

"""
    quadweights(xs::AbstractVector, order::Int) -> Vector{Float64}

Composite Newton-Cotes quadrature weights for arbitrary node positions `xs`,
using panels of `order+1` points exact for polynomials of degree `order`.

Weights may be negative for high `order` or non-uniform `xs`. For new code,
prefer `grid(M, l, h, dist)` which returns points and weights together using
a rule matched to the grid distribution.

# Arguments
- `xs`: Mesh points. Need not be sorted; decreasing inputs are handled internally.
- `order`: Polynomial degree of the local rule (1 = trapezoidal, 2 = Simpson, …).
"""
function quadweights(xs::AbstractVector, order::Int)
    xs_sorted = issorted(xs) ? xs : reverse(xs)

    N  = length(xs_sorted)
    ws = zeros(N)

    ii = 1
    while ii < N
        ie = ii + order
        ie = N - ie < order ? N : ie
        ws[ii:ie] .+= _quadweights(view(xs_sorted, ii:ie))
        ii = ie
    end

    issorted(xs) || reverse!(ws)

    return ws
end


"""
    _quadweights(xs::AbstractVector) -> Vector{Float64}

Exact quadrature weights for a single panel of nodes `xs` by solving the
Vandermonde system for monomial exactness up to degree `length(xs)-1`:

    A w = b,    A[d+1, i] = xs[i]^d,    b[d+1] = (xs[end]^{d+1} - xs[1]^{d+1}) / (d+1)

For two nodes returns the trapezoidal weights `[h, h]/2`. For three equally-spaced
nodes returns Simpson weights `[1, 4, 1]*h/3`.

Weights may be negative for large or non-uniform panels.
"""
function _quadweights(xs::AbstractVector)
    N = length(xs)
    b = [(xs[end]^(d + 1) - xs[1]^(d + 1)) / (d + 1) for d in 0:N-1]
    A = [xs[i]^d for d in 0:N-1, i in 1:N]
    return A \ b
end
