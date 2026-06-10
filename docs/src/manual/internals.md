# Internal Layout and Kernels

This page is for users who want to understand how `FDGrids.jl` works internally
or contribute to the package. The main design goal is simple:

> Store only the coefficients that are actually used, but keep the storage
> regular enough that applying the operator is fast and predictable.

The two central types are [`DiffMatrix`](@ref) and [`AdjointDiffMatrix`](@ref).
They use different coefficient layouts because applying a forward
finite-difference matrix row-by-row is different from applying its transpose.

## Type Parameters and Invariants

The concrete forward type is

```julia
DiffMatrix{T, WIDTH, OPTIMISE}
```

where:

- `T` is the coefficient element type,
- `WIDTH` is the odd stencil width,
- `OPTIMISE` controls whether compact LU stores the diagonal of `U` inverted.

The stencil width is a type parameter, not a runtime field. This is what lets
generated kernels emit a fixed number of multiply-add operations for each
interior stencil. The main invariants are:

1. `WIDTH` is odd and at least `3`.
2. `length(D.coeffs) == size(D, 1) * WIDTH`.
3. Each logical row of a `DiffMatrix` owns exactly `WIDTH` stored entries.
4. Entries outside a row's stored stencil are structural zeros.

The adjoint type is

```julia
AdjointDiffMatrix{T, WIDTH}
```

and stores a reference to the parent `DiffMatrix` plus an output-major
coefficient vector for the transposed action. Unlike `DiffMatrix`, the adjoint
storage has variable-length boundary rows.

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

## Indexing and Mutation

Scalar indexing reconstructs the dense interpretation on demand. For a logical
entry `(i, j)`, the implementation computes the row-local stencil coordinate
`m`. If `m` is between `1` and `WIDTH`, the corresponding stored coefficient is
returned; otherwise the value is `zero(T)`.

`setindex!` uses the same mapping. Assignments inside the stored stencil mutate
the compact coefficient vector. Assignments outside the stored stencil are
ignored, because the compact layout has no storage location for them. This is
important when imposing boundary conditions: overwriting a row can only change
entries already represented by that row's band.

For example, the first row of a width-5 matrix stores columns `1:5`, so
`D[1, 1] = 1` is stored but `D[1, 10] = 1` is ignored. This keeps the matrix
layout fixed and predictable, but it means `setindex!` cannot widen the band.

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

## Generated Loop Structure

The generated multiplication code is assembled from small expression builders:

- `_make_ref` emits array indexing expressions with the differentiated index
  replaced by a stencil expression.
- `_make_kernel_fixed` emits an unrolled fixed-width dot product.
- `_make_kernel_variable` emits the short variable-length adjoint boundary
  dot product.
- `_make_loop_expr_forward` and `_make_loop_expr_adjoint` wrap those kernels in
  the loop nest for the requested array rank and differentiated dimension.

Loop ordering is chosen so dimensions before `DIM` are nested inside the
differentiated loop, while dimensions after `DIM` are outside it. Conceptually,
for rank `N`, every non-`DIM` coordinate identifies one fiber, and the
generated code applies the same one-dimensional operator to that fiber.

The forward body is the simplest region:

```julia
base = i - HWIDTH
y[...] = dot(A.coeffs[ptr:ptr+WIDTH-1], x[base:base+WIDTH-1, ...])
ptr += WIDTH
```

The adjoint body has the same fixed-width structure. The adjoint head and tail
are different because the number of contributing forward rows grows and shrinks
near the boundaries. Those regions use variable-length inner loops and more
pointer arithmetic. This is why the adjoint benchmark is close to, but usually
slightly slower than, the forward benchmark when boundary work is visible.

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

Boundary regions are classified in global rows and then translated back to
local indices. This matters for a middle slab: its first stored or owned row is
not a physical boundary merely because it is local index `1`.

Two storage conventions satisfy the contract:

- A dense local buffer can include ghost rows in its ordinary axes and use a
  shifted `local_rng` for the owned rows.
- A halo-aware array can keep its ordinary axes for owned rows and provide
  scalar indices outside those axes, such as `0` and `n + 1`, for ghost cells.

In both cases, callers must populate the required ghost values before applying
the operator.

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

## Adjoint Construction

The adjoint coefficient builder loops over adjoint outputs `j`, which are
columns of the parent forward matrix. For each output, it visits only the
forward rows that can contain column `j`. The stored coefficient is the dense
transpose entry, optionally scaled for a weighted inner product:

```math
A^+_{j,i} = D_{i,j}\frac{w_i}{w_j}.
```

For the unweighted adjoint, `w_i/w_j = 1`. For the weighted adjoint, the
scaling is applied once during construction, so `mul!` does not need to touch
the weight vector.

The construction requires `N > 2*WIDTH` because the adjoint storage layout has
a head, a body, and a tail. Without a body region the pointer formulas and
generated multiplication assumptions are not valid.

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

## Compact LU Storage

The compact linear-solve path reuses the `DiffMatrix` coefficient vector for
the LU factors. After `lu!(D)`, the stored band no longer represents a
differentiation matrix:

- entries below the diagonal inside the band store the multipliers of the
  unit-lower triangular factor `L`,
- the diagonal and upper-band entries store `U`,
- if `OPTIMISE=true`, the diagonal entries of `U` are replaced by their
  reciprocals.

The generated triangular solvers in `src/linalg.jl` specialize on `WIDTH`.
Their interior loops directly index the compact coefficient vector and unroll
the known number of subdiagonal or superdiagonal updates. Boundary rows use
ordinary indexing because the available band width changes near the matrix
ends.

This compact solve path deliberately does not pivot. Pivoting would require row
interchanges and potentially different fill patterns, which do not fit the
fixed row layout. The pivoted reference path therefore converts to LAPACK's
banded workspace and stores a pivot vector separately.

## GPU Kernels

The CUDA extension reuses the host coefficient layouts verbatim. The
`DiffMatrix` and `AdjointDiffMatrix` types are parametric over their backing
vector type, so the only thing that changes on the GPU is that
`coeffs::CuArray` instead of `coeffs::Vector`. The forward and adjoint kernels
operate on those same row-major and output-major layouts described above.

### Parallel decomposition

Both GPU kernels use a one-thread-per-output design. Each thread receives a
flat 0-based index `idx`, decomposes it into 1-based cartesian coordinates
`i_1, …, i_N` using the array shape, and applies the stencil by varying the
coordinate along `DIM`. The decomposition expression is identical to a
column-major linearisation, so consecutive threads walk the first dimension
first.

All index arithmetic uses `Int32`, matching the native register width on
NVIDIA hardware. The total element count per launch is therefore capped at
`2^31`, which is comfortably above any practical single-GPU array size.

### Forward kernel

The forward GPU kernel is the GPU analogue of the host body kernel: for
output `i` along `DIM`, it loads `WIDTH` coefficients from
`A.coeffs[(i-1)WIDTH + 1 .. i WIDTH]` and `WIDTH` input values starting at a
boundary-aware base index, then writes the unrolled dot product to `y`. The
dot product is fully unrolled at generation time because `WIDTH` is a type
parameter.

Boundary handling is the same three-region split as the host kernel:

  - **head** `i ≤ HWIDTH`           → `base = 1`,
  - **body** `HWIDTH < i ≤ M-HWIDTH` → `base = i - HWIDTH`,
  - **tail** `i > M - HWIDTH`        → `base = M - WIDTH + 1`.

Warp divergence is confined to threads in the boundary slabs along `DIM`,
which is at most `2 HWIDTH ≤ WIDTH` threads per fiber.

### Adjoint kernel

The adjoint GPU kernel mirrors the host adjoint logic. The body region is
fully unrolled like the forward kernel, because every body output has exactly
`WIDTH` contributing forward rows and starts at `start = j - HWIDTH`. The
head and tail regions use small runtime loops bounded by `WIDTH + HWIDTH`
because their row lengths grow and shrink with `j`.

Pointer formulas are reused unchanged. For output `j` along `DIM`, the
kernel computes:

```math
\mathrm{ptr}(j) =
\begin{cases}
1 + \mathrm{HWIDTH}(j-1) + (j-1)j/2,
& j \le \mathrm{WIDTH},\\
(j-1)\mathrm{WIDTH} + 1,
& \mathrm{WIDTH} < j \le M-\mathrm{WIDTH},\\
(M-\mathrm{WIDTH})\mathrm{WIDTH} + 1
+ (\mathrm{WIDTH}+\mathrm{HWIDTH}+1)(j_t-1)
- (j_t-1)j_t/2,
& j > M-\mathrm{WIDTH},
\end{cases}
```

with `j_t = j - (M - WIDTH)`. These match the host-side closed forms
inside `_ptr_for_j`.

Because weights are baked into `A.coeffs` at construction time, weighted and
unweighted adjoints dispatch to the exact same kernel; the kernel never sees
the weight vector.

### Launch and adaption

The host-side `mul!` methods are responsible for validation, picking a launch
configuration with `launch_configuration`, and invoking the kernel through
`@cuda`. The `@cuda` macro automatically adapts each closure argument: a
`DiffMatrix{T, WIDTH, OPTIMISE, CuArray}` becomes
`DiffMatrix{T, WIDTH, OPTIMISE, CuDeviceArray}` so the kernel reads
device-side pointers. The `Adapt.adapt_structure` methods in the extension
make this work without changing any of the host code paths.

The same `Adapt.adapt(CuArray, _)` machinery is what users invoke to perform
host-to-device transfers in their own code. The extension's `cu` overloads
are a thin wrapper that goes through `CUDA.cu` to obtain a Float32 storage,
preserving the convention of `cu` on plain arrays.

## Source-Code Guide

The implementation is split by responsibility:

- `src/utils.jl`: finite-difference weights and coefficient-table assembly.
- `src/diffmatrix.jl`: compact forward matrix type and dense/indexing views.
- `src/adjoint.jl`: adjoint storage layout, pointer formulas, and construction.
- `src/matmul.jl`: generated kernels for forward and adjoint application.
- `src/linalg.jl`: reference banded routines and compact `DiffMatrix` solves.
- `src/grids.jl`: grid constructors and quadrature weights.
- `ext/FDGridsCUDAExt.jl`: GPU kernels and the `Adapt` / `cu` machinery.

When extending the package, keep these invariants in mind:

1. `DiffMatrix` stores exactly `WIDTH` coefficients per row.
2. `WIDTH` is a type parameter and must remain available to generated kernels.
3. `AdjointDiffMatrix` stores output-major variable-length rows.
4. `global_idx` always refers to local index `1` in decomposed-domain kernels.
5. `full(A)` and scalar indexing should agree with `mul!` semantics, including
   weighted adjoints.
6. Adding device backends should preserve the row-major (forward) and
   output-major variable-length (adjoint) coefficient layouts so the kernels
   can stay shared with the existing implementations.
