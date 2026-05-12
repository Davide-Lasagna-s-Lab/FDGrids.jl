# Benchmarks

The scripts in `benchmarks/` measure the two performance-critical operations
in FDGrids: linear solves and matrix–vector products.  Run them with Julia 1.11
or later; each script saves an SVG to `docs/src/assets/benchmarks/`.

```bash
# One-time setup (from the package root)
julia --project=benchmarks -e '
    using Pkg
    Pkg.develop(path=".")
    Pkg.instantiate()'

julia --project=benchmarks benchmarks/linsolve.jl
julia --project=benchmarks benchmarks/matmul.jl
julia --project=benchmarks benchmarks/adjoint.jl
```

The figures embed the CPU model and available RAM of the machine they were run on.

## Linear solve

Compares three solve paths on a 1D grid, sweeping over grid sizes N = 16 … 2048
and stencil widths 3, 5, 7.  Each timing is the **median** of 300 samples.

| Path | Implementation |
|------|----------------|
| **LAPACK** | `lu(D)` → `DiffMatrixLU` → `ldiv!` via `dgbtrf!`/`dgbtrs!` (pivoted banded LU) |
| **FDGrids** | `lu!(copy(D))` → `@generated ldiv!` (compile-time-unrolled, no pivoting) |
| **Generic** | same in-place factorisation, dispatched through plain scalar loops |

The left panel shows absolute solve time; the right panel shows the speedup of
the FDGrids path over LAPACK.

```@raw html
<img src="../../assets/benchmarks/linsolve_speedup.svg"
     alt="Linear solve speedup"
     style="width:100%; max-width:900px;" />
```

## Matrix–vector product

Benchmarks `mul!(y, D, x, Val(dim))` across:

- Array dimensions N = 1, 2, 3, 4 (square arrays, same size along every axis)
- Differentiation directions `dim` = 1 … N
- Stencil widths 3, 5, 7

To reduce the influence of thermal throttling and cache-state bias, all
`(N, width, dim, size)` combinations are shuffled into a random order and the
full sweep is repeated 10 times.  The **median** over those 10 runs is plotted.

The x-axis shows the per-axis array size (e.g. 64² means a 64×64 array).
Throughput is reported in **GEl/s** (giga-elements per second): total array
elements divided by median wall time.  Higher is better.  Line colour encodes
stencil width; line style encodes the differentiation direction.

```@raw html
<img src="../../assets/benchmarks/matmul.svg"
     alt="mul! throughput"
     style="width:100%; max-width:1100px;" />
```

## Forward vs adjoint

Measures the speedup of `mul!(y, adjoint(D), x, Val(dim))` relative to the
forward `mul!(y, D, x, Val(dim))`, defined as **forward time / adjoint time**.
A value above 1 means the adjoint is faster; below 1 it carries overhead.
A dashed line at 1 marks parity.

The same array dimensions, stencil widths, and size ranges as the matrix–vector
product benchmark are used, with the same randomised 10-pass median methodology.
Colour encodes stencil width; line style encodes differentiation direction.

```@raw html
<img src="../../assets/benchmarks/adjoint.svg"
     alt="Adjoint speedup over forward"
     style="width:100%; max-width:1100px;" />
```
