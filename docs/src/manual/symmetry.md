# Boundary Symmetry

By default the boundary rows of a [`DiffMatrix`](@ref) use one-sided stencils of
the full width. When the differentiated field has a known parity about a
boundary — it is even or odd under reflection across that wall — those one-sided
rows can be replaced with a *centred* stencil whose out-of-domain nodes are
mirrored back into the grid. This recovers centred-difference accuracy at the
boundary and exactly encodes the parity (for example, a zero normal derivative
for an even field).

This is a lightweight, per-row modification. It rewrites only the boundary rows
of the active sides; interior rows are never touched. It is **not** a full
boundary-condition system and **not** pipe-axis regularity — it is a mirror
stencil and nothing more.

## Symmetry Types

A boundary side is described by a [`Symmetry`](@ref) object:

| Type | Reflection | Effect |
|------|-----------|--------|
| [`NoSymmetry`](@ref) | — | side keeps its one-sided boundary rows |
| [`EvenSymmetry`](@ref)`(c)` | ``u(2c - x) = u(x)`` | even mirror about centre `c` |
| [`OddSymmetry`](@ref)`(c)` | ``u(2c - x) = -u(x)`` | odd mirror about centre `c` |

The reflection is written as the mirror of an absolute coordinate,
``x \mapsto 2c - x``, because that is exactly what the stencil computes — a ghost
node at ``x`` maps to the real node at ``2c - x``. Measuring displacement from
the centre instead, this is the familiar ``u(c - x) = u(c + x)`` (even) and
``u(c - x) = -u(c + x)`` (odd); the two forms are identical under
``x \to c + x``.

The `symmetry` keyword of `DiffMatrix` takes a `(left, right)` tuple of these
objects, one per boundary:

```@example symmetry
using FDGrids
using LinearAlgebra

xs = grid(48, -1, 1, GaussLobattoGrid()).xs

# even mirror on the left boundary, ordinary one-sided rows on the right
D = DiffMatrix(xs, 5, 1; symmetry = (EvenSymmetry(xs[1]), NoSymmetry()))
symmetry(D)
```

The default — equivalent to omitting the keyword — is
`(NoSymmetry(), NoSymmetry())`.

## How the Mirror Stencil Works

For a stencil of width `w` the half-width is `HWIDTH = w ÷ 2`, so an active left
side rewrites rows `1 … HWIDTH` and an active right side rewrites rows
`N-HWIDTH+1 … N`. Each such row `i` is rebuilt from the virtual centred stencil
spanning `i-HWIDTH … i+HWIDTH`:

- in-domain nodes (`1 ≤ m ≤ N`) are ordinary grid points;
- out-of-domain nodes are **ghost** points reflected about the centre `c`. A
  ghost at `x` maps to the real node at `2c - x`, and its finite-difference
  weight is folded back onto that real column with sign `+1` for
  [`EvenSymmetry`](@ref) and `-1` for [`OddSymmetry`](@ref).

When the centre coincides with the boundary node (the usual case, `c = xs[1]` or
`c = xs[end]`) the node on the wall is not duplicated by its own reflection.

## Accuracy at the Boundary

Take a field that is genuinely even about the left wall, `u(x) = cos(π(x+1))`,
whose exact derivative is `u'(x) = -π sin(π(x+1))`. Comparing the plain
one-sided operator with the even-mirror operator on the boundary rows (`1:2` for
a width-5 stencil):

```@example symmetry
u  = cos.(π .* (xs .+ 1))          # even about x = -1
ux = -π .* sin.(π .* (xs .+ 1))    # exact derivative

Dplain = DiffMatrix(xs, 5, 1)
Deven  = DiffMatrix(xs, 5, 1; symmetry = (EvenSymmetry(xs[1]), NoSymmetry()))

err_plain = maximum(abs, (Dplain * u .- ux)[1:2])
err_even  = maximum(abs, (Deven  * u .- ux)[1:2])
(err_plain, err_even)
```

The mirror stencil reduces the boundary-row error by more than an order of
magnitude. It also encodes the parity exactly: for an even field the normal
derivative at the wall vanishes, so the first row of `Deven` returns ~0:

```@example symmetry
(Deven * u)[1]      # ≈ 0, the exact value of u'(-1)
```

An odd field is mirrored with the opposite sign. For `v(x) = sin(π(x+1))`, odd
about `x = -1`:

```@example symmetry
v  = sin.(π .* (xs .+ 1))          # odd about x = -1
vx = π .* cos.(π .* (xs .+ 1))     # exact derivative

Dodd = DiffMatrix(xs, 5, 1; symmetry = (OddSymmetry(xs[1]), NoSymmetry()))

err_plain = maximum(abs, (Dplain * v .- vx)[1:2])
err_odd   = maximum(abs, (Dodd   * v .- vx)[1:2])
(err_plain, err_odd)
```

## Centre Placement

The centre is **required** and must be a `Real`. For boundary-centred symmetry
use `xs[1]` on the left or `xs[end]` on the right. For an active side the centre
must lie at or beyond the boundary node, so that the reflected ghost nodes fall
outside the grid:

- left side: `c ≤ xs[1]`
- right side: `c ≥ xs[end]`

A centre placed inside the grid makes the mirror ambiguous and is rejected:

```@example symmetry
try
    DiffMatrix(xs, 5, 1; symmetry = (EvenSymmetry(0.0), NoSymmetry()))
catch err
    err
end
```

## Both Boundaries

The two entries of the tuple are independent, so each wall can carry its own
parity. A field that is even on the left and odd on the right, for instance:

```@example symmetry
Dboth = DiffMatrix(xs, 5, 1;
                   symmetry = (EvenSymmetry(xs[1]), OddSymmetry(xs[end])))
(symmetry_left(Dboth), symmetry_right(Dboth))
```

## Inspecting a Symmetric Operator

The attached symmetry travels with the operator and can be queried with
[`symmetry`](@ref), [`symmetry_left`](@ref), [`symmetry_right`](@ref), and
[`centre`](@ref):

```@example symmetry
(symmetry_left(Deven), centre(symmetry_left(Deven)))
```

Because the parity is baked into the stored coefficients, adjoints and
quadrature-weighted adjoints (see [Adjoints](adjoints.md)) operate on the
already-mirrored rows with no extra bookkeeping.

For signatures, see the [API Reference](../api.md#Boundary-Symmetry).
