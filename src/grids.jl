export AbstractGridDistribution,
       MappedGrid,
       UniformGrid,
       GaussLobattoGrid,
       grid,
       gridpoints   # retained for backward compatibility

# ================================================================================
# Grid distribution types
# ================================================================================

"""
    AbstractGridDistribution

Abstract supertype for all grid distributions. Concrete subtypes dispatch the
`grid` function to the appropriate node-placement and quadrature algorithm,
always returning nodes and weights together so they are guaranteed to be consistent.

| Type | Nodes | Endpoints | Quadrature | Always positive weights? |
|---|---|---|---|---|
| `MappedGrid(α, order)` | Mapped Chebyshev | yes | Composite Newton-Cotes | not guaranteed |
| `UniformGrid()` | Equally spaced | yes | Composite trapezoidal | yes |
| `GaussLobattoGrid()` | Chebyshev-Lobatto | yes | Clenshaw-Curtis | yes |
"""
abstract type AbstractGridDistribution end


"""
    MappedGrid(α=0.5, order=4)

Mapped Chebyshev grid with clustering parameter `α ∈ (0,1]`. Nodes are placed at

    xⱼ = asin(-α cos(πj/(M-1))) / asin(α) · (h-l)/2 + (h+l)/2,    j = 0, …, M-1

`α=1` gives uniform spacing; `α→0` approaches the Chebyshev-Lobatto distribution.

`order` is the polynomial degree of the composite Newton-Cotes quadrature rule.
Positive weights are not guaranteed for high `order` or strongly non-uniform nodes.
Typical values: `1` (trapezoidal), `2` (Simpson), `3`, `4`.
"""
struct MappedGrid <: AbstractGridDistribution
    α    ::Float64
    order::Int
    function MappedGrid(α::Real = 0.5, order::Int = 4)
        0 < α ≤ 1 || throw(ArgumentError("α must ∈ (0, 1]"))
        order ≥ 1  || throw(ArgumentError("order must be ≥ 1"))
        return new(Float64(α), order)
    end
end


"""
    UniformGrid()

Equally-spaced grid. The associated quadrature is the composite trapezoidal rule,
which always produces strictly positive weights.
"""
struct UniformGrid <: AbstractGridDistribution end


"""
    GaussLobattoGrid()

Chebyshev-Lobatto (Gauss-Lobatto) grid: `M` nodes at

    xⱼ = (l+h)/2 + (h-l)/2 · cos(π(M-1-j)/(M-1)),    j = 0, …, M-1

including both endpoints, clustered near the boundaries.

The associated quadrature is **Clenshaw-Curtis**, which has strictly positive weights
and is spectrally accurate for smooth integrands.

This is the recommended distribution when a positive-definite inner product is required
(e.g. for `weighted_adjoint`). Fields with zero boundary conditions produce zero endpoint
contributions regardless of the endpoint weights, so Gauss-Lobatto is equally suitable
for zero-BC problems while supporting the standard BC-enforcement pattern of overwriting
rows 1 and `M` of `D`.
"""
struct GaussLobattoGrid <: AbstractGridDistribution end


# ================================================================================
# Main API: grid(M, l, h, dist) -> (xs, ws)
# ================================================================================

"""
    grid(M, l=-1.0, h=1.0, dist=MappedGrid(0.5)) -> (xs::Vector, ws::Vector)

Compute `M` grid points and their associated quadrature weights on `[l, h]`
according to the distribution `dist`. Points and weights are always computed
together, ensuring they are guaranteed to be consistent.

# Arguments
- `M`: Number of grid points. Must be ≥ 2.
- `l`, `h`: Left and right endpoints; must satisfy `l < h`.
- `dist`: An `AbstractGridDistribution`. Defaults to `MappedGrid(0.5)`.

# Returns
A named tuple `(xs = points, ws = weights)` where:
- `xs[i]` is the `i`-th grid point in ascending order.
- `ws[i]` is the quadrature weight at `xs[i]`, so that `sum(f.(xs) .* ws)`
  approximates `∫_l^h f(x) dx`.

# Examples
```julia
xs, ws = grid(64, -1, 1, GaussLobattoGrid())  # Clenshaw-Curtis, always positive
xs, ws = grid(64, -1, 1, UniformGrid())       # trapezoidal, always positive
xs, ws = grid(64, -1, 1, MappedGrid(0.5, 2))  # Simpson composite, α=0.5
xs, ws = grid(64, -1, 1, MappedGrid(0.5))     # Newton-Cotes order 4, α=0.5
```
"""
function grid(M::Int, l::Real = -1.0, h::Real = 1.0,
              dist::AbstractGridDistribution = MappedGrid(0.5))
    M > 1 || throw(ArgumentError("M must be ≥ 2"))
    l < h  || throw(ArgumentError("l must be less than h"))
    return _grid(M, Float64(l), Float64(h), dist)
end


# ================================================================================
# Internal implementations
# ================================================================================

# ---- MappedGrid ----

function _grid(M::Int, l::Float64, h::Float64, g::MappedGrid)
    j  = 0:M-1
    xs = asin.(-g.α .* cos.(π .* j ./ (M - 1))) ./ asin(g.α) .* (h - l) / 2 .+ (h + l) / 2
    xs = collect(xs)
    ws = _newton_cotes_weights(xs, g.order)
    return (xs = xs, ws = ws)
end


# ---- UniformGrid ----

function _grid(M::Int, l::Float64, h::Float64, ::UniformGrid)
    xs = collect(range(l, h; length = M))
    h_ = (h - l) / (M - 1)
    ws    = fill(h_, M)
    ws[1] = h_ / 2
    ws[M] = h_ / 2
    return (xs = xs, ws = ws)
end


# ---- GaussLobattoGrid ----

function _grid(M::Int, l::Float64, h::Float64, ::GaussLobattoGrid)
    xs = [(l + h) / 2 + (h - l) / 2 * cos(π * (M - 1 - j) / (M - 1))
          for j in 0:M-1]
    ws = _clenshaw_curtis_weights(M, l, h)
    return (xs = xs, ws = ws)
end


# ================================================================================
# Quadrature weight implementations
# ================================================================================

"""
    _clenshaw_curtis_weights(M, l, h) -> Vector

Clenshaw-Curtis weights for `M` Chebyshev-Lobatto nodes on `[l, h]`.

Uses the explicit DCT-based formula (Waldvogel 2006):

    c_k = 2 / (1 - (2k)²),    k = 0, 1, …, ⌊(M-1)/2⌋

    w_j = 2/(M-1) · (c₀/2 + Σ_{k=1}^{⌊(M-1)/2⌋} c_k cos(2πjk/(M-1)))

Weights are scaled by `(h-l)/2` and endpoint weights are halved. Always strictly
positive.

# Reference
Waldvogel, J. (2006). Fast construction of the Fejér and Clenshaw-Curtis
quadrature rules. *SIAM J. Sci. Comput.*, 46(1), 195–202.
"""
function _clenshaw_curtis_weights(M::Int, l::Float64, h::Float64)
    scale = (h - l) / 2
    K     = (M - 1) ÷ 2
    c     = [2.0 / (1 - (2k)^2) for k in 0:K]

    ws = zeros(M)
    for j in 0:M-1
        s = c[1] / 2
        for k in 1:K
            s += c[k + 1] * cos(2π * j * k / (M - 1))
        end
        ws[j + 1] = s * 2 / (M - 1)
    end

    ws[1] /= 2
    ws[M] /= 2

    return scale .* ws
end


"""
    _newton_cotes_weights(xs, order) -> Vector

Composite Newton-Cotes quadrature weights for the node vector `xs` using panels
of `order+1` points. Delegates to `_quadweights_panel` on each panel.

Weights may be negative for high `order` or non-uniform nodes.
"""
function _newton_cotes_weights(xs::AbstractVector, order::Int)
    N  = length(xs)
    ws = zeros(N)
    ii = 1
    while ii < N
        ie = ii + order
        ie = N - ie < order ? N : ie
        ws[ii:ie] .+= _quadweights_panel(view(xs, ii:ie))
        ii = ie
    end
    return ws
end


"""
    _quadweights_panel(xs) -> Vector

Exact quadrature weights for a single panel of nodes `xs` by solving the
Vandermonde system for monomial exactness up to degree `length(xs)-1`:

    A w = b,    A[d+1, i] = xs[i]^d,    b[d+1] = (xs[end]^{d+1} - xs[1]^{d+1}) / (d+1)

Returns trapezoidal weights for two nodes, Simpson weights for three equally-spaced
nodes. Weights may be negative for large or non-uniform panels.
"""
function _quadweights_panel(xs::AbstractVector)
    N = length(xs)
    b = [(xs[end]^(d + 1) - xs[1]^(d + 1)) / (d + 1) for d in 0:N-1]
    A = [xs[i]^d for d in 0:N-1, i in 1:N]
    return A \ b
end


# ================================================================================
# Backward compatibility
# ================================================================================

"""
    gridpoints(M, l=-1.0, h=1.0, α=0.5) -> Vector

Return grid points only. Preserved for backward compatibility.
For new code, prefer `grid(M, l, h, dist)` which returns both points and weights.
"""
function gridpoints(M::Int, l::Real = -1.0, h::Real = 1.0, α::Real = 0.5)
    xs, _ = grid(M, l, h, MappedGrid(α))
    return xs
end

"""
    gridpoints(M, l, h, dist::AbstractGridDistribution) -> Vector

Return grid points only for the given distribution.
For new code, prefer `grid(M, l, h, dist)` which returns both points and weights.
"""
function gridpoints(M::Int, l::Real, h::Real, dist::AbstractGridDistribution)
    xs, _ = grid(M, l, h, dist)
    return xs
end
