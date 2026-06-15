export AbstractGridDistribution,
       MappedGrid,
       UniformGrid,
       GaussLobattoGrid,
       ChebyshevGrid,
       grid

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
| `ChebyshevGrid()` | Chebyshev-Lobatto | yes | Clenshaw-Curtis | yes |
"""
abstract type AbstractGridDistribution end


"""
    MappedGrid(α=0.5, order=2)

Mapped Chebyshev grid with clustering parameter `α ∈ (0,1]`. Nodes are placed at

    xⱼ = asin(-α cos(πj/(M-1))) / asin(α) · (h-l)/2 + (h+l)/2,    j = 0, …, M-1

`α=1` gives uniform spacing; `α→0` approaches the Chebyshev-Lobatto distribution.

`order` is the polynomial degree of the composite Newton-Cotes quadrature rule.
Positive weights are not guaranteed for high `order` or strongly non-uniform nodes.
The default is `order=2`, i.e. Simpson panels. Other typical values are
`1` (trapezoidal), `3`, and `4`.

# Examples
```julia
g = grid(48, -1, 1, MappedGrid(0.5, 4))
sum(exp.(g.xs) .* g.ws)
```
"""
struct MappedGrid <: AbstractGridDistribution
    α    ::Float64
    order::Int
    function MappedGrid(α::Real = 0.5, order::Int = 2)
        0 < α ≤ 1 || throw(ArgumentError("α must ∈ (0, 1]"))
        order ≥ 1  || throw(ArgumentError("order must be ≥ 1"))
        return new(Float64(α), order)
    end
end


"""
    UniformGrid()

Equally-spaced grid. The associated quadrature is the composite trapezoidal rule,
which always produces strictly positive weights.

# Examples
```julia
g = grid(16, 0, 1, UniformGrid())
```
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
(e.g. for `adjoint(D, w)`). Fields with zero boundary conditions produce zero endpoint
contributions regardless of the endpoint weights, so Gauss-Lobatto is equally suitable
for zero-BC problems while supporting the standard BC-enforcement pattern of overwriting
rows 1 and `M` of `D`.

# Examples
```julia
g = grid(32, -1, 1, GaussLobattoGrid())
D = DiffMatrix(g.xs, 5, 1)
```
"""
struct GaussLobattoGrid <: AbstractGridDistribution end


"""
    ChebyshevGrid()

Chebyshev-Lobatto grid on `[l, h]`: `M` nodes at

    x_j = (l+h)/2 + (h-l)/2 * sin(pi * (2j - M + 1) / (2(M-1))),
          j = 0, ..., M-1

including both endpoints and clustered near the boundaries. The associated
quadrature is Clenshaw-Curtis.

# Examples
```julia
g = grid(64, -1, 1, ChebyshevGrid())
```
"""
struct ChebyshevGrid <: AbstractGridDistribution end


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
g = grid(64, -1, 1, GaussLobattoGrid())  # Clenshaw-Curtis, always positive
sum(exp.(g.xs) .* g.ws)

xs, ws = grid(64, -1, 1, UniformGrid())  # named tuples can be destructured
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

"""
    _grid(M, l, h, g::MappedGrid) -> (xs, ws)

Internal implementation for `MappedGrid`.

The nodes are generated in ascending order on `[l,h]` and weights are computed
with composite Newton-Cotes panels of degree `g.order`.
"""
function _grid(M::Int, l::Float64, h::Float64, g::MappedGrid)
    j  = 0:M-1
    xs = asin.(-g.α .* cos.(π .* j ./ (M - 1))) ./ asin(g.α) .* (h - l) / 2 .+ (h + l) / 2
    xs = collect(xs)
    ws = _newton_cotes_weights(xs, g.order)
    return (xs = xs, ws = ws)
end


# ---- UniformGrid ----

"""
    _grid(M, l, h, ::UniformGrid) -> (xs, ws)

Internal implementation for equally spaced nodes on `[l,h]`.

Weights are the composite trapezoidal rule: half weights at the endpoints and
uniform full weights in the interior.
"""
function _grid(M::Int, l::Float64, h::Float64, ::UniformGrid)
    xs = collect(range(l, h; length = M))
    h_ = (h - l) / (M - 1)
    ws    = fill(h_, M)
    ws[1] = h_ / 2
    ws[M] = h_ / 2
    return (xs = xs, ws = ws)
end


# ---- GaussLobattoGrid ----

"""
    _grid(M, l, h, ::GaussLobattoGrid) -> (xs, ws)

Internal implementation for Chebyshev-Lobatto nodes on `[l,h]`.

The nodes are ordered from left to right and paired with Clenshaw-Curtis
quadrature weights from `_clenshaw_curtis_weights`.
"""
function _grid(M::Int, l::Float64, h::Float64, ::GaussLobattoGrid)
    xs = [(l + h) / 2 + (h - l) / 2 * cos(π * (M - 1 - j) / (M - 1))
          for j in 0:M-1]
    ws = _clenshaw_curtis_weights(M, l, h)
    return (xs = xs, ws = ws)
end


# ---- ChebyshevGrid ----

function _grid(M::Int, l::Float64, h::Float64, ::ChebyshevGrid)
    xs = [(l + h) / 2 + (h - l) / 2 * sin(π * (2j - M + 1) / (2 * (M - 1)))
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
quadrature rules. *BIT Numerical Mathematics*, 46, 195–202.
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

The points in `xs` should be sorted in ascending order.

Weights may be negative for high `order` or non-uniform nodes.
"""
function _newton_cotes_weights(xs::AbstractVector, order::Int)
    issorted(xs) || throw(ArgumentError("Newton-Cotes weights require sorted grid points"))

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
