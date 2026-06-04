# Boundary symmetry for `DiffMatrix`.
#
# A boundary's symmetry is described by a `Symmetry` object: `NoSymmetry()`
# leaves that side untouched, while `Even(c)`/`Odd(c)` rewrite that side's
# boundary rows with a centred stencil whose out-of-range nodes are mirrored
# about the centre `c` (see `apply_symmetry_stencil!`). The `symmetry` field of
# a `DiffMatrix` holds a `(left, right)` tuple of these objects and is the main
# switch; interior rows and `NoSymmetry()` sides are left untouched.

"""
    Symmetry

Abstract supertype for boundary symmetries. Concrete types are `NoSymmetry`,
`Even`, and `Odd`.
"""
abstract type Symmetry end

"""
    NoSymmetry()

No boundary symmetry: the side keeps its one-sided boundary rows unchanged.
"""
struct NoSymmetry <: Symmetry end

"""
    Even(centre = nothing)

Even (`u(2c - x) = u(x)`) boundary symmetry about `centre`. `centre` is a `Real`
or `nothing`, where `nothing` means the default boundary node (`xs[1]` on the
left, `xs[end]` on the right).
"""
struct Even{C<:Union{Real, Nothing}} <: Symmetry
    centre :: C
end
Even() = Even(nothing)

"""
    Odd(centre = nothing)

Odd (`u(2c - x) = -u(x)`) boundary symmetry about `centre`. `centre` is a `Real`
or `nothing`, where `nothing` means the default boundary node (`xs[1]` on the
left, `xs[end]` on the right).
"""
struct Odd{C<:Union{Real, Nothing}} <: Symmetry
    centre :: C
end
Odd() = Odd(nothing)

const NO_SYMMETRY = (NoSymmetry(), NoSymmetry())

# The point a side is mirrored about; `nothing` means "use the default boundary
# node". `NoSymmetry` carries no centre.
centre(::NoSymmetry) = nothing
centre(s::Even) = s.centre
centre(s::Odd)  = s.centre

# Sign applied to a reflected (ghost) node's finite-difference weight.
ghost_sign(::Even) = 1.0
ghost_sign(::Odd)  = -1.0

function validate_symmetry(symmetry)
    symmetry isa Tuple && length(symmetry) == 2 ||
        throw(ArgumentError("symmetry must be a (left, right) tuple"))

    all(s -> s isa Symmetry, symmetry) ||
        throw(ArgumentError("symmetry entries must be NoSymmetry, Even, or Odd"))

    return symmetry
end

symmetry(D) = D.symmetry
symmetry_left(D)  = D.symmetry[1]
symmetry_right(D) = D.symmetry[2]

# The resolved (recorded) centre of each side; `nothing` when the side carries
# no explicit centre (`NoSymmetry`, or `Even()`/`Odd()` left at the default).
symmetry_centre(D) = (centre(D.symmetry[1]), centre(D.symmetry[2]))
symmetry_centre_left(D)  = centre(D.symmetry[1])
symmetry_centre_right(D) = centre(D.symmetry[2])

# Resolve the centre for one side: `nothing` means the default boundary node.
_resolve_centre(c, default) = c === nothing ? float(default) : float(c)

"""
    apply_symmetry_stencil!(C, xs, width, order, symmetry) -> C

Rewrite the boundary rows of the compact coefficient matrix `C` (size
`(width, N)`, column `i` is row `i`'s stencil) so that active boundaries use a
centred stencil mirrored about their symmetry centre.

`symmetry` is the `(left, right)` main switch: a side that is `NoSymmetry()` is
never touched. For an active side, each boundary row `i` (the left rows
`i ≤ HWIDTH`, the right rows `i > N - HWIDTH`) is rebuilt from the virtual
centred stencil `m = i-HWIDTH : i+HWIDTH`. Real indices `1 ≤ m ≤ N` are ordinary
grid nodes; out-of-range `m` are ghost nodes reflected about the centre `c`:

  - `Even`: `u(2c - x) =  u(x)`
  - `Odd` : `u(2c - x) = -u(x)`

A ghost node maps back to a real column `j` (`xs[j]`), and its finite-difference
weight is folded onto column `j` with sign `+1` (even) or `-1` (odd). Returns `C`
unchanged when `symmetry == NO_SYMMETRY`.

This is a lightweight mirror stencil: it only rewrites the active sides' boundary
rows in place, never building ghost-point objects or touching interior rows. It
is not a full boundary-condition system and not pipe-specific regularity.
"""
function apply_symmetry_stencil!(C, xs, width::Int, order::Int, symmetry)
    symmetry == NO_SYMMETRY && return C

    N      = length(xs)
    HWIDTH = width >> 1
    symL, symR = symmetry

    # Guard against a centre placed inside the grid, which would make the mirror
    # stencil ambiguous. Only checked for active sides.
    if !(symL isa NoSymmetry)
        cL = _resolve_centre(centre(symL), first(xs))
        cL ≤ first(xs) ||
            throw(ArgumentError("left symmetry centre must satisfy c ≤ first(xs)"))
        for i in 1:HWIDTH
            _mirror_row!(C, xs, width, order, i, symL, cL, N, HWIDTH)
        end
    end
    if !(symR isa NoSymmetry)
        cR = _resolve_centre(centre(symR), last(xs))
        cR ≥ last(xs) ||
            throw(ArgumentError("right symmetry centre must satisfy c ≥ last(xs)"))
        for i in (N - HWIDTH + 1):N
            _mirror_row!(C, xs, width, order, i, symR, cR, N, HWIDTH)
        end
    end
    return C
end

# Rebuild a single boundary row `i` in place using the mirrored centred stencil.
function _mirror_row!(C, xs, width::Int, order::Int, i::Int, sym::Symmetry, c::Real, N::Int, HWIDTH::Int)
    left  = clamp(i - HWIDTH, 1, N - width + 1)   # first stored column of this row
    nodes = Vector{Float64}(undef, width)
    cols  = Vector{Int}(undef, width)
    signs = Vector{Float64}(undef, width)

    gs = ghost_sign(sym)
    # `isapprox` avoids brittle exact float comparison when deciding whether the
    # centre sits on the boundary node (which skips duplicating that node).
    on_left  = isapprox(c, first(xs))
    on_right = isapprox(c, last(xs))

    k = 0
    for m in (i - HWIDTH):(i + HWIDTH)
        k += 1
        if 1 ≤ m ≤ N
            nodes[k] = xs[m];  cols[k] = m;  signs[k] = 1.0
        elseif m < 1
            # left ghost: skip the centre node itself when it sits on x₁
            j = on_left ? 2 - m : 1 - m
            nodes[k] = 2c - xs[j];  cols[k] = j;  signs[k] = gs
        else
            # right ghost: skip the centre node itself when it sits on x_N
            j = on_right ? 2N - m : 2N + 1 - m
            nodes[k] = 2c - xs[j];  cols[k] = j;  signs[k] = gs
        end
    end

    w = get_weights(float(xs[i]), nodes, order + 2)

    @views C[:, i] .= 0
    for k in 1:width
        s = cols[k] - left + 1
        C[s, i] += signs[k] * w[k, order + 1]
    end
    return C
end
