export gridpoints, quadweights, _quadweights

# The primary API for grid points and quadrature weights is `grid(M, l, h, dist)`
# in grids.jl, which returns both together. The functions below are retained only
# for backward compatibility with existing code that uses the old split APIs.

"""
    gridpoints(M, l=-1.0, h=1.0, α=0.5) -> Vector

Return grid points only. Preserved for backward compatibility.

Deprecated: use `grid(M, l, h, MappedGrid(α)).xs` instead. Use `grid` directly
when both points and quadrature weights are needed.
"""
function gridpoints(M::Int, l::Real = -1.0, h::Real = 1.0, α::Real = 0.5)
    Base.depwarn("gridpoints is deprecated; use grid(M, l, h, MappedGrid(α)).xs instead", :gridpoints)

    xs, _ = grid(M, l, h, MappedGrid(α))
    return xs
end

"""
    gridpoints(M, l, h, dist::AbstractGridDistribution) -> Vector

Return grid points only for the given distribution.

Deprecated: use `grid(M, l, h, dist).xs` instead. Use `grid` directly when both
points and quadrature weights are needed.
"""
function gridpoints(M::Int, l::Real, h::Real, dist::AbstractGridDistribution)
    Base.depwarn("gridpoints is deprecated; use grid(M, l, h, dist).xs instead", :gridpoints)

    xs, _ = grid(M, l, h, dist)
    return xs
end


"""
    quadweights(xs::AbstractVector, order::Int) -> Vector{Float64}

Composite Newton-Cotes quadrature weights for arbitrary node positions `xs`,
using panels of `order+1` points exact for polynomials of degree `order`.

Weights may be negative for high `order` or non-uniform `xs`. For new code,
prefer `grid(M, l, h, dist).ws`, or use `grid(M, l, h, dist)` directly when
both points and weights are needed.

Deprecated: use `grid(M, l, h, dist).ws` instead.

# Arguments
- `xs`: Mesh points. Need not be sorted; decreasing inputs are handled internally.
- `order`: Polynomial degree of the local rule (1 = trapezoidal, 2 = Simpson, …).
"""
function quadweights(xs::AbstractVector, order::Int)
    Base.depwarn("quadweights is deprecated; use grid(M, l, h, dist).ws instead", :quadweights)

    xs_sorted = issorted(xs) ? xs : reverse(xs)

    N  = length(xs_sorted)
    ws = zeros(N)

    ii = 1
    while ii < N
        ie = ii + order
        ie = N - ie < order ? N : ie
        ws[ii:ie] .+= _quadweights_impl(view(xs_sorted, ii:ie))
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

Deprecated: use `grid(M, l, h, dist).ws` for quadrature weights associated with
a supported grid distribution.
"""
function _quadweights(xs::AbstractVector)
    Base.depwarn("_quadweights is deprecated; use grid(M, l, h, dist).ws instead", :_quadweights)
    return _quadweights_impl(xs)
end

function _quadweights_impl(xs::AbstractVector)
    N = length(xs)
    b = [(xs[end]^(d + 1) - xs[1]^(d + 1)) / (d + 1) for d in 0:N-1]
    A = [xs[i]^d for d in 0:N-1, i in 1:N]
    return A \ b
end
