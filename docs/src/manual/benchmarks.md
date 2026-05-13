# Benchmarks

The scripts in `benchmarks/` measure the two performance-critical operations in
`FDGrids.jl`: compact linear solves and dimension-wise multiplication. Each
script writes an SVG under `docs/src/assets/benchmarks/`.

```bash
# One-time setup from the package root
julia --project=benchmarks -e '
    using Pkg
    Pkg.develop(path=".")
    Pkg.instantiate()'

julia --project=benchmarks benchmarks/linsolve.jl
julia --project=benchmarks benchmarks/matmul.jl
julia --project=benchmarks benchmarks/adjoint.jl
```

The figures embed the CPU model and available RAM of the machine used to run
the benchmark. Treat absolute timings as machine-specific; use trends and
relative comparisons to understand implementation tradeoffs.

## Linear Solve

This benchmark compares the three solve paths documented in
[Linear Solves](linear-solves.md):

| Path | Implementation |
|------|----------------|
| **LAPACK** | `lu(D)` to `DiffMatrixLU`, then `ldiv!` via `gbtrf!`/`gbtrs!` with pivoting. |
| **FDGrids** | `lu!(copy(D))`, then generated compact `ldiv!`, no pivoting. |
| **Generic** | Same no-pivot banded algorithm using ordinary scalar indexing. |

The sweep uses grid sizes `N = 16:2048` and stencil widths `3`, `5`, and `7`.
Each timing is the median of 300 samples.

The left panel shows absolute solve time. The right panel shows speedup of the
compact FDGrids path over LAPACK. The comparison is not only an algorithmic
comparison: LAPACK pivots and uses its own banded workspace, while the compact
path avoids conversion and keeps factors inside `DiffMatrix` storage.

```@raw html
<img src="../../assets/benchmarks/linsolve_speedup.svg"
     alt="Linear solve speedup"
     style="width:100%; max-width:900px;" />
```

## Matrix-Vector Product

This benchmark measures `mul!(y, D, x, Val(dim))` across:

- array ranks `1`, `2`, `3`, and `4`,
- square arrays with the same size along every axis,
- differentiation directions `dim = 1:rank`,
- stencil widths `3`, `5`, and `7`.

To reduce thermal and cache-order bias, all `(rank, width, dim, size)`
combinations are shuffled and the full sweep is repeated 10 times. The plotted
value is the median over those passes.

Throughput is reported in **GEl/s**: total array elements divided by median wall
time. Higher is better. Color encodes stencil width; line style encodes the
differentiation direction.

```@raw html
<img src="../../assets/benchmarks/matmul.svg"
     alt="mul! throughput"
     style="width:100%; max-width:1100px;" />
```

## Forward vs Adjoint

This benchmark compares `mul!(y, adjoint(D), x, Val(dim))` with the forward
operator. The plotted value is

```math
\frac{\text{forward time}}{\text{adjoint time}}.
```

A value above `1` means the adjoint is faster; a value below `1` means the
adjoint is slower. The dashed line marks parity.

The adjoint should be close to the forward path because the centered body loop
uses the same fixed-width generated stencil. A small overhead is expected near
boundaries: adjoint head and tail rows have variable-length compact storage,
extra pointer arithmetic, and a short runtime loop. The effect is most visible
when many short fibers are differentiated, because each fiber pays the boundary
cost.

```@raw html
<img src="../../assets/benchmarks/adjoint.svg"
     alt="Adjoint speedup over forward"
     style="width:100%; max-width:1100px;" />
```
