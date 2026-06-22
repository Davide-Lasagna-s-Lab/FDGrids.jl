# GPU Support

`FDGrids.jl` ships an optional CUDA extension that runs forward and adjoint
`DiffMatrix` applications on NVIDIA GPUs. The extension is a thin layer over
the CPU types: the same `DiffMatrix` and `AdjointDiffMatrix` objects are used
on both devices, only the backing coefficient storage changes.

The extension is loaded automatically as soon as `using CUDA` happens after
the package has been loaded. No explicit import is needed.

```julia
using FDGrids
using CUDA          # triggers the FDGridsCUDAExt extension
using LinearAlgebra
```

## What Is Supported

The extension provides GPU implementations of:

- `mul!(y, D::DiffMatrix, x, Val(DIM))` — forward operator along any axis.
- `mul!(y, A::AdjointDiffMatrix, x, Val(DIM))` — adjoint operator along any axis.
- `cu(D)` and `cu(adjoint(D))` — host-to-device transfer (Float64 → Float32).
- `Adapt.adapt(CuArray, D)` — host-to-device transfer preserving precision.

What is **not** supported on the GPU:

- `lu!`, `ldiv!`, and any of the banded solve routines in `linalg.jl`.
- The 6-argument distributed `mul!` entry that takes `global_idx` and
  `local_rng`. Decomposed-domain GPU applications remain a future addition.
- Scalar indexing (`D[i, j]`, `full(D)`, etc.) on device-side operators. Use
  `Array(D.coeffs)` to bring coefficients back to the host first.

These are excluded because they require pivoting, multi-device communication,
or CPU-style scalar access that does not map cleanly to a single-kernel
implementation.

## Transferring Operators to the Device

Two transfer paths are provided. Both produce a `DiffMatrix` (or
`AdjointDiffMatrix`) whose coefficient vector lives on the GPU. The choice
between them is a precision decision:

| Transfer | Element type | Use it when |
|----------|--------------|-------------|
| `cu(D)` | downcasts to `Float32` | Single-precision is acceptable; matches the convention of `CUDA.cu` on plain arrays. |
| `Adapt.adapt(CuArray, D)` | preserves the host element type | Double precision is required, e.g. tight tolerances or condition-sensitive solves. |

```julia
using FDGrids, CUDA, Adapt, LinearAlgebra

xs = collect(range(-1, 1; length = 256))
D  = DiffMatrix(xs, 5, 1)                  # Float64 on host

D_f32 = cu(D)                              # Float32 on device
D_f64 = Adapt.adapt(CuArray, D)            # Float64 on device
```

The adjoint type works the same way:

```julia
A    = adjoint(D)
A_g  = cu(A)                               # Float32 adjoint on device
```

The parent matrix is moved alongside the adjoint, so `adjoint(A_g)`
structurally returns the GPU forward operator without a second copy:

```julia
adjoint(A_g) === A_g.parent                # true
```

## Applying the Forward Operator

The GPU `mul!` is dispatched automatically whenever the operator's
coefficient backing is a `CuArray`. CPU and GPU code therefore look
identical:

```julia
g  = grid(256, -1, 1, GaussLobattoGrid())
D  = DiffMatrix(g.xs, 5, 1)
Dg = cu(D)

u  = CuArray(Float32.(sin.(g.xs)))
du = similar(u)
mul!(du, Dg, u)                            # GPU dispatch

# bring back for inspection
Array(du)[1:3]
```

For higher-dimensional arrays, choose the differentiated axis with `Val`:

```julia
nx, ny = 256, 32
Dx     = cu(DiffMatrix(grid(nx, -1, 1).xs, 5, 1))

u2  = CuArray(rand(Float32, nx, ny))
ux2 = similar(u2)
mul!(ux2, Dx, u2, Val(1))                  # differentiate along axis 1
```

## Applying the Adjoint

The adjoint dispatch mirrors the forward one. Both ordinary and weighted
adjoints go through the same GPU kernel because the weights are baked into
`A.coeffs` at construction time:

```julia
g   = grid(256, -1, 1, GaussLobattoGrid())
D   = DiffMatrix(g.xs, 5, 1)
Aw  = adjoint(D, g.ws)                     # weighted adjoint, built on CPU
Awg = cu(Aw)                               # transfer to device

v   = CuArray(Float32.(randn(length(g.xs))))
y   = similar(v)
mul!(y, Awg, v)
```

Build the adjoint on the host, then `cu` (or `Adapt.adapt`) the result. The
adjoint constructor uses host-side scalar indexing internally, so building it
directly from a GPU `DiffMatrix` is not supported.

## Tuning the Launch Configuration

Both GPU `mul!` methods accept an `nthreads` keyword that overrides the
per-block thread count. The auto-tuned default — obtained from
`CUDA.launch_configuration` — is appropriate for production use; the keyword
is intended for benchmarks that need to compare different occupancies:

```julia
mul!(yg, Dg, ug)                           # auto-tuned
mul!(yg, Dg, ug; nthreads = 128)           # explicit block size
```

The result of the auto-tuned `mul!` is stored for any remaining calls to `mul!`
with the same argument types, persistent for the lifetime of the Julia session.
To reset this cached result, call `FDGrids.reset_launch_params()`, which will
prompt `mul!` to perform auto-tuning the next time it is called.

The result of the multiplication does not depend on the block size, only the
runtime.

## Limitations and Caveats

- **Element-type uniformity.** `y`, `x`, and `A.coeffs` must share an element
  type. Mismatches will promote silently inside the kernel and produce
  type-unstable code.
- **Single-precision is the default for `cu`.** This matches the convention
  of `CUDA.cu` on plain arrays. For Float64, use `Adapt.adapt(CuArray, _)`.
- **Per-launch element cap.** Index arithmetic is done in `Int32` for
  efficiency, which limits a single launch to roughly `2^31` elements. This
  is far above any practical single-GPU array.
- **No GPU LU.** The compact `lu!` / `ldiv!` path in `linalg.jl` is CPU-only.
  Repeated solves with the same operator should be performed on the host.

## Verifying Against the CPU

For development or correctness checks, run the same operator on both devices
and compare. The CPU path is exercised by the package test suite and acts as
the trusted reference:

```julia
y_cpu = similar(x_cpu)
mul!(y_cpu, D, x_cpu)

x_gpu = CuArray(Float32.(x_cpu))
y_gpu = similar(x_gpu)
mul!(y_gpu, cu(D), x_gpu)

Array(y_gpu) ≈ Float32.(y_cpu)             # within Float32 rtol
```

## Running the GPU Test Suite

The CUDA tests are off by default because most contributors do not have a
CUDA-capable device. The runner opts in with a positional `gpu_ext`
argument:

```bash
julia --project=test test/runtests.jl gpu_ext
```

With that argument:

- The base tests still run, unless they are skipped with `skip_base`.
- The runner loads `CUDA.jl` and inspects `CUDA.functional()`. If the
  runtime is missing, GPU tests are skipped with a warning rather than
  failing, so CI without a GPU continues to pass.
- When CUDA is functional, the file
  [`test/FDGridsCUDAExt/test_gpu.jl`](https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl/tree/master/test/FDGridsCUDAExt)
  runs the type-adaptation, forward, adjoint, weighted-adjoint, and launch
  validation testsets.

To run only the GPU tests on a CUDA-capable workstation, combine both
flags:

```bash
julia --project=test test/runtests.jl skip_base gpu_ext
```

For storage layouts and kernel structure, see
[Internal Layout and Kernels](internals.md#GPU-Kernels).
