# Internal Layout and Kernels

This page is for users who want to understand how `FDGrids.jl` works internally
or contribute to the package. The main design goal is simple:

> Store only the coefficients that are actually used, but keep the storage
> regular enough that applying the operator is fast and predictable.

The two central types are [`DiffMatrix`](@ref) and [`AdjointDiffMatrix`](@ref).
They use different coefficient layouts because applying a forward
finite-difference matrix row-by-row is different from applying its transpose.

## Forward Operator Layout

A `DiffMatrix` represents an `N × N` banded finite-difference matrix with a
fixed stencil width `WIDTH`. Instead of storing the dense matrix, it stores
exactly `WIDTH` coefficients per row:

```math
\mathrm{length}(D.\mathrm{coeffs}) = N\,\mathrm{WIDTH}.
```

The coefficients for logical row `i` occupy:

```math
p_i = (i-1)\,\mathrm{WIDTH} + 1,
```

so the stored slice is:

```julia
D.coeffs[p_i : p_i + WIDTH - 1]
```

The logical dense row is reconstructed by placing those coefficients on the
stencil selected for row `i`. For an interior row with
`HWIDTH = WIDTH ÷ 2`, the stencil starts at:

```math
i - \mathrm{HWIDTH}.
```

Near the left boundary, the stencil is shifted right. Near the right boundary,
it is shifted left. Therefore the start column is:

```math
\mathrm{left}(i) =
\mathrm{clamp}(i-\mathrm{HWIDTH},\,1,\,N-\mathrm{WIDTH}+1).
```

The dense interpretation is:

```math
D_{i,\mathrm{left}(i)+k-1} = D.\mathrm{coeffs}_{p_i+k-1},
\qquad k = 1,\ldots,\mathrm{WIDTH}.
```

All other entries in row `i` are structural zeros.

## Example: Width 5

For `WIDTH = 5`, each row stores five coefficients. The stencil positions look
like this:

```text
row 1:  x x x x x . . . .
row 2:  x x x x x . . . .
row 3:  x x x x x . . . .
row 4:  . x x x x x . . .
row 5:  . . x x x x x . .
...
row N-2: . . . . x x x x x
row N-1: . . . . x x x x x
row N:   . . . . x x x x x
```

The compact coefficient vector stores these rows back-to-back:

```text
row 1              row 2              row 3
| c11 c12 c13 c14 c15 | c21 c22 c23 c24 c25 | ...
```

This row-major compact layout is what makes the forward multiplication simple:
for output row `i`, read `WIDTH` coefficients, read `WIDTH` input values, take a
dot product, and advance the coefficient pointer by `WIDTH`.

## Forward Matrix Multiplication

For vectors, applying the operator is conceptually:

```julia
for i in 1:N
    left = clamp(i - HWIDTH, 1, N - WIDTH + 1)
    ptr  = (i - 1) * WIDTH + 1
    y[i] = sum(D.coeffs[ptr + k - 1] * x[left + k - 1]
               for k in 1:WIDTH)
end
```

The real implementation generalizes this to arrays of any rank and any
differentiated dimension:

```julia
mul!(y, D, x, Val(DIM))
```

For each fiber along `DIM`, it applies the same row-wise stencil. The generated
kernel splits the differentiated index into three regions:

- **head**: rows that use the left boundary stencil,
- **body**: rows that use centered stencils,
- **tail**: rows that use the right boundary stencil.

The body is especially cheap: the input base index is `i - HWIDTH`, and the
coefficient pointer advances by exactly `WIDTH` after every output. Because
`WIDTH` is a type parameter, the inner dot product is unrolled at code
generation time.

## Why Generated Kernels?

The package uses generated functions in `src/matmul.jl` so that the generated
loop nest knows:

- the rank `N` of the array,
- the differentiated dimension `DIM`,
- the stencil width `WIDTH`.

That lets the code emit direct indexing expressions like:

```julia
x[i1, i2, i3]
```

with only the differentiated index replaced by the stencil index. This avoids
runtime construction of Cartesian indices and keeps the inner stencil loop
small.

For example, differentiating a 3D array along dimension 2 conceptually emits
loops of the form:

```julia
for i3 in axes(x, 3)
    for i2 in rows_to_compute
        for i1 in axes(x, 1)
            y[i1, i2, i3] = stencil_dot(...)
        end
    end
end
```

The actual generated expression also handles head/body/tail regions and
decomposed-domain local ranges.

## Decomposed-Domain Pointers

The lower-level multiplication method:

```julia
mul!(y_local, D, x_local, Val(DIM), global_idx, local_rng)
```

uses `global_idx` to map local index `1` to a global matrix row. The global row
computed at local index `i_local` is:

```math
i_\mathrm{global} = \mathrm{global\_idx} + i_\mathrm{local} - 1.
```

The coefficient pointer for the first computed local row is therefore:

```math
(\mathrm{global\_idx} + i_\mathrm{local} - 2)\,\mathrm{WIDTH} + 1.
```

This is why local arrays with halo points should pass `global_idx` for local
index `1`, not for `first(local_rng)`.

## Adjoint Operator Layout

The transpose of a banded matrix cannot use the same simple row-major
`WIDTH`-per-row layout near the boundaries. A column of the forward matrix may
receive fewer or more nonzero contributions depending on its position.

`AdjointDiffMatrix` therefore stores coefficients output-by-output for the
adjoint action:

```math
y_j = \sum_i A^*_{j,i} x_i.
```

The number of coefficients for adjoint output `j` is:

- `j + HWIDTH` in the head region,
- `WIDTH` in the body region,
- `N - j + HWIDTH + 1` in the tail region.

For `WIDTH = 5`, this looks like:

```text
output 1:  coefficients from forward rows 1:3
output 2:  coefficients from forward rows 1:4
output 3:  coefficients from forward rows 1:5
output 4:  coefficients from forward rows 2:6
...
output N-2: coefficients from forward rows N-4:N
output N-1: coefficients from forward rows N-3:N
output N:   coefficients from forward rows N-2:N
```

The helper `_ptr_for_j(j, N, Val(WIDTH))` returns the first coefficient for
output `j` in this variable-length storage.

## Adjoint Pointer Formula

Let `HWIDTH = WIDTH ÷ 2`. The pointer has three cases:

```math
\mathrm{ptr}(j) =
\begin{cases}
1 + \mathrm{HWIDTH}(j-1) + (j-1)j/2,
& j \le \mathrm{WIDTH},\\
(j-1)\mathrm{WIDTH} + 1,
& \mathrm{WIDTH} < j \le N-\mathrm{WIDTH},\\
(N-\mathrm{WIDTH})\mathrm{WIDTH} + 1
+ (\mathrm{WIDTH}+\mathrm{HWIDTH}+1)(j_t-1)
- (j_t-1)j_t/2,
& j > N-\mathrm{WIDTH},
\end{cases}
```

where `j_t = j - (N - WIDTH)`.

The head formula sums the growing row lengths. The body formula works because
all previous body outputs have exactly `WIDTH` coefficients. The tail formula
sums the shrinking row lengths.

## Weighted Adjoint Coefficients

For a weighted inner product with `W = Diagonal(w)`, the weighted adjoint is:

```math
D^+ = W^{-1}D^T W.
```

The stored coefficient for adjoint output `j` and source row `i` is:

```math
D_{i,j}\frac{w_i}{w_j}.
```

This factor is inserted once when constructing `adjoint(D, w)`. Multiplication
then uses the same adjoint kernel as the unweighted case.

## Source-Code Guide

The implementation is split by responsibility:

- `src/utils.jl`: finite-difference weights and coefficient-table assembly.
- `src/diffmatrix.jl`: compact forward matrix type and dense/indexing views.
- `src/adjoint.jl`: adjoint storage layout, pointer formulas, and construction.
- `src/matmul.jl`: generated kernels for forward and adjoint application.
- `src/linalg.jl`: reference banded routines and compact `DiffMatrix` solves.
- `src/grids.jl`: grid constructors and quadrature weights.

When extending the package, keep these invariants in mind:

1. `DiffMatrix` stores exactly `WIDTH` coefficients per row.
2. `WIDTH` is a type parameter and must remain available to generated kernels.
3. `AdjointDiffMatrix` stores output-major variable-length rows.
4. `global_idx` always refers to local index `1` in decomposed-domain kernels.
5. `full(A)` and scalar indexing should agree with `mul!` semantics, including
   weighted adjoints.

