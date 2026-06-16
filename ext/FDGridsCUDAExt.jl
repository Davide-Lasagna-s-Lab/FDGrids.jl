module FDGridsCUDAExt

using CUDA
using Adapt
using LinearAlgebra

using CUDA: i32

using FDGrids
using FDGrids: DiffMatrix, AdjointDiffMatrix


# ================================================================================
# Overview
# ================================================================================
#
# This extension makes `DiffMatrix` and `AdjointDiffMatrix` operate on the GPU.
# Loading it is automatic: as soon as a user does `using CUDA` while FDGrids is
# already loaded, Julia activates this extension, which adds three things:
#
#   1. `Adapt.adapt_structure` methods so the existing types can be transparently
#      moved through Adapt — both for host-side transfers and for the
#      CuArray → CuDeviceArray adaption that happens automatically during @cuda
#      kernel launches.
#
#   2. `CUDA.cu` overloads matching the host-array convention (Float64 → Float32
#      downcast). For precision-preserving transfer, use `Adapt.adapt(CuArray, _)`.
#
#   3. `LinearAlgebra.mul!` methods specialised to a `<:CuArray` coefficient
#      backing. CPU code paths are not touched.
#
# No `lu!`, `ldiv!`, or banded solve is provided on the GPU. Repeated linear
# solves remain a CPU-only path.
#
# Indices inside the device kernels are Int32 (the native register width on
# NVIDIA hardware). This caps the total per-launch element count at 2^31 ≈ 2.1G,
# which is well above any practical single-GPU array.


# ================================================================================
# Adapt support
# ================================================================================

"""
    Adapt.adapt_structure(to, d::DiffMatrix) -> DiffMatrix

Adapt a `DiffMatrix` for the given adaptor.

This method serves two distinct callers:

  - Host transfers like `Adapt.adapt(CuArray, D)`, which transfer the
    coefficient storage to a `CuArray` while preserving the original element
    type. This is the precision-preserving counterpart to `cu`.
  - The CUDA kernel adaptor used internally by `@cuda`, which transforms
    `CuArray` → `CuDeviceArray` so the closure captures device-side pointers.

In both cases, the element type of the returned `DiffMatrix` is taken from the
adapted coefficient vector. This keeps the wrapper consistent with adaptors
that change precision (`cu` casts Float64 → Float32) without forcing CPU code
paths to know anything about device types.
"""
function Adapt.adapt_structure(to, d::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE}
    # Adapt the coefficient vector first. For the CuArray adaptor this hits the
    # generic Adapt fallback (Vector → CuArray of the same eltype); for the
    # @cuda KernelAdaptor it transforms CuArray → CuDeviceArray. We re-read the
    # eltype afterwards so precision-changing adaptors do not produce a
    # type-inconsistent wrapper.
    coeffs = Adapt.adapt(to, d.coeffs)
    return DiffMatrix{eltype(coeffs), WIDTH, OPTIMISE}(coeffs, d.symmetry)
end

"""
    Adapt.adapt_structure(to, d::AdjointDiffMatrix) -> AdjointDiffMatrix

Adapt an `AdjointDiffMatrix` for the given adaptor.

The parent `DiffMatrix` and the adjoint's own coefficient vector are adapted
independently. The two vectors do not share storage on either device — adjoint
construction builds its own output-major coefficient vector — so adapting them
separately keeps that invariant.

The outer `AdjointDiffMatrix(parent, coeffs)` constructor checks that the
adapted element types agree, which they always do for uniform adaptors like
`CuArray` or the CUDA kernel adaptor.
"""
function Adapt.adapt_structure(to, d::AdjointDiffMatrix)
    parent = Adapt.adapt(to, d.parent)
    coeffs = Adapt.adapt(to, d.coeffs)
    return AdjointDiffMatrix(parent, coeffs)
end


# ================================================================================
# Convenience GPU transfer with `cu`
# ================================================================================

"""
    CUDA.cu(d::DiffMatrix) -> DiffMatrix

Transfer a `DiffMatrix` to the GPU, downcasting Float64 to Float32.

This matches the convention of `CUDA.cu` on plain arrays: a single-precision
transfer that is convenient for inference workloads and small-precision
solvers. For precision-preserving transfer, use [`Adapt.adapt(CuArray, D)`](@ref).

# Examples
```julia
using CUDA, FDGrids, LinearAlgebra

xs = range(-1, 1; length = 64)
D  = DiffMatrix(xs, 5, 1)       # Float64 on host

D_gpu = cu(D)                   # Float32 on device
typeof(D_gpu).parameters[1]     # → Float32
```
"""
function CUDA.cu(d::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE}
    # `Array` first so a non-`Vector` coefficient store (e.g. another wrapped
    # array) is normalised to a plain host vector before `cu` performs the
    # precision-changing transfer.
    coeffs = CUDA.cu(Array(d.coeffs))
    return DiffMatrix{eltype(coeffs), WIDTH, OPTIMISE}(coeffs, d.symmetry)
end

"""
    CUDA.cu(d::AdjointDiffMatrix) -> AdjointDiffMatrix

Transfer an `AdjointDiffMatrix` to the GPU, downcasting Float64 to Float32.

The parent matrix and the precomputed adjoint coefficients are both
transferred. The parent is retained so that `adjoint(D_gpu_adjoint)` still
returns the GPU-side forward operator without an extra copy.

# Examples
```julia
using CUDA, FDGrids, LinearAlgebra

xs   = range(-1, 1; length = 64)
D    = DiffMatrix(xs, 5, 1)
A    = adjoint(D)
A_g  = cu(A)

adjoint(A_g) === A_g.parent     # → true
```
"""
function CUDA.cu(d::AdjointDiffMatrix)
    parent = CUDA.cu(d.parent)
    coeffs = CUDA.cu(Array(d.coeffs))
    return AdjointDiffMatrix(parent, coeffs)
end


# ================================================================================
# GPU kernel helpers
# ================================================================================
#
# Both forward and adjoint kernels share a one-thread-per-output design. Each
# thread receives a flat 0-based index `idx`, decomposes it into 1-based
# cartesian coordinates `i_1, …, i_N`, and applies the stencil by varying the
# coordinate along DIM. All index arithmetic is in Int32 to match the native
# register width on NVIDIA hardware.

"""
    _build_decomp(N) -> Expr

Build the AST that decomposes the flat thread index `idx` into N-dimensional
cartesian indices `i_1, …, i_N`.

For each dimension `d`, the snippet emits:

```julia
i_d  = (rem_ % sz[d]) + 1i32
rem_ = rem_ ÷ sz[d]
```

starting from `rem_ = idx`. The resulting indices follow Julia's column-major
layout, so consecutive threads walk dimension 1 first. The expression is
spliced into the generated kernel body, where `idx` and `sz` are already in
scope.
"""
function _build_decomp(N::Int)
    decomp = quote
        rem_ = idx
    end
    for d in 1:N
        push!(decomp.args, :($(Symbol(:i_, d)) = (rem_ % sz[$d]) + 1i32))
        push!(decomp.args, :(rem_ = rem_ ÷ sz[$d]))
    end
    return decomp
end

"""
    _x_idx(DIM, N, kexpr) -> Expr

Construct the index tuple for an N-dimensional array access whose index along
`DIM` is `kexpr` and whose other indices are the per-thread cartesian indices
`i_1, …, i_N`.

This mirrors the role of `_make_ref` in `src/matmul.jl`: it builds the
indexing expression for the differentiated axis at generation time, so the
unrolled stencil body indexes the array with concrete index expressions
instead of constructing CartesianIndex objects on the device.
"""
_x_idx(DIM::Integer, N::Integer, kexpr) =
    Expr(:tuple, ntuple(d -> d == DIM ? kexpr : Symbol(:i_, d), N)...)


# ================================================================================
# Forward kernel
# ================================================================================

"""
    _gpu_forward_kernel!(y, A_coeffs, x, sz, ::Val{DIM}, ::Val{WIDTH})

Apply the forward `DiffMatrix` operator along dimension `DIM`.

Each thread computes one output element. The flat thread id is decomposed
into N-D cartesian indices, then the stencil dot product is evaluated using
`WIDTH` consecutive coefficients of `A_coeffs` and `WIDTH` consecutive entries
of `x` whose first index along `DIM` (the "base") depends on the boundary
region:

  - **head** `i ≤ HWIDTH`             → `base = 1`
  - **body** `HWIDTH < i ≤ M - HWIDTH` → `base = i - HWIDTH`
  - **tail** `i > M - HWIDTH`           → `base = M - WIDTH + 1`

The dot product is fully unrolled at generation time because `WIDTH` is a
type parameter. The coefficient row for output `i` always starts at index
`(i-1)*WIDTH + 1` in `A_coeffs`, so the unrolled body emits direct loads
without per-iteration pointer arithmetic.

# Arguments
- `y`: output array, written one element per thread.
- `A_coeffs`: flat coefficient vector of length `M*WIDTH` (row-major).
- `x`: input array with the same shape as `y`.
- `sz`: the array shape as an `NTuple{N, Int32}`.
- `Val{DIM}`: dimension to differentiate along (1-based).
- `Val{WIDTH}`: stencil width (odd, ≥ 3).

This function should not be called directly. The host-side `mul!` method
takes care of launching it with an appropriate block / grid configuration.
"""
@generated function _gpu_forward_kernel!(y, A_coeffs, x,
                                         sz::NTuple{N, Int32},
                                         ::Val{DIM},
                                         ::Val{WIDTH}) where {N, DIM, WIDTH}
    HWIDTH = WIDTH >> 1
    Wi32   = Int32(WIDTH)
    Hi32   = Int32(HWIDTH)

    iDIM    = Symbol(:i_, DIM)
    out_idx = Expr(:tuple, ntuple(d -> Symbol(:i_, d), N)...)
    decomp  = _build_decomp(N)

    # Build the unrolled WIDTH-tap dot product as `s = c[ptr]*x[base] + …`,
    # parameterised by the runtime `base` that the kernel computes once per
    # thread. Each subsequent tap advances both the coefficient pointer and
    # the input index by one. WIDTH is a generated-time constant, so the loop
    # body is laid out at compile time and the device executes a fixed-length
    # straight-line dot product.
    accumulator = quote
        ptr = (i - 1i32) * $Wi32 + 1i32
        s   = A_coeffs[ptr] * x[$(_x_idx(DIM, N, :base))...]
    end
    for p in 1:(WIDTH - 1)
        pi32 = Int32(p)
        push!(accumulator.args,
              :(s += A_coeffs[ptr + $pi32] *
                     x[$(_x_idx(DIM, N, :(base + $pi32)))...]))
    end

    return quote
        # Flat 0-based thread id; threads past the end return without writing.
        idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x - 1i32
        idx ≥ prod(sz) && return nothing

        # Decompose into 1-based cartesian indices i_1, …, i_N.
        $decomp

        # `i` is the row of the operator that this thread computes; `M` is the
        # number of operator rows (= size of x along DIM).
        i = $iDIM
        M = sz[$DIM]

        # Boundary-aware stencil base. Branching here is unavoidable, but only
        # the first/last HWIDTH outputs along DIM diverge from the centered
        # case, so warp divergence is confined to thin boundary slabs.
        base = i ≤ $Hi32     ? 1i32             :
               i > M - $Hi32 ? M - $Wi32 + 1i32 :
               i - $Hi32

        @inbounds begin
            $accumulator
            y[$out_idx...] = s
        end
        return nothing
    end
end


# ================================================================================
# Adjoint kernel
# ================================================================================

"""
    _gpu_adjoint_kernel!(y, A_coeffs, x, sz, ::Val{DIM}, ::Val{WIDTH})

Apply the transposed `AdjointDiffMatrix` operator along dimension `DIM`.

The adjoint coefficient layout has variable-length boundary rows. For output
index `j` along `DIM`, the stencil reads:

  - **head**, `j ≤ WIDTH`:           `start = 1`,        `len = j + HWIDTH`
  - **body**, `WIDTH < j ≤ M-WIDTH`: `start = j - HWIDTH`, `len = WIDTH`
  - **tail**, `j > M - WIDTH`:        `start = j - HWIDTH`, `len = M - j + HWIDTH + 1`

with pointers into `A_coeffs` given in closed form by `_ptr_for_j`. The body
case is the hot path and is fully unrolled, exactly like the forward kernel.
Head and tail use small runtime loops bounded by `WIDTH + HWIDTH`; warp
divergence is confined to the first and last `WIDTH` outputs along DIM.

# Arguments
Identical in role to [`_gpu_forward_kernel!`](@ref): `y`, `A_coeffs`, `x`,
`sz`, `Val{DIM}`, `Val{WIDTH}`. The kernel should not be called directly —
the host-side `mul!` method takes care of launch configuration.
"""
@generated function _gpu_adjoint_kernel!(y, A_coeffs, x,
                                         sz::NTuple{N, Int32},
                                         ::Val{DIM},
                                         ::Val{WIDTH}) where {N, DIM, WIDTH}
    HWIDTH = WIDTH >> 1
    Wi32   = Int32(WIDTH)
    Hi32   = Int32(HWIDTH)

    iDIM    = Symbol(:i_, DIM)
    out_idx = Expr(:tuple, ntuple(d -> Symbol(:i_, d), N)...)
    decomp  = _build_decomp(N)

    # ----- Body: fully unrolled WIDTH-tap sum, identical pattern to the
    # forward kernel. Coefficient row for output j starts at (j-1)*WIDTH + 1,
    # because the body has WIDTH-wide rows just like the parent matrix.
    body_block = quote
        ptr = (j - 1i32) * $Wi32 + 1i32
        s   = A_coeffs[ptr] * x[$(_x_idx(DIM, N, :(j - $Hi32)))...]
    end
    for p in 1:(WIDTH - 1)
        pi32 = Int32(p)
        push!(body_block.args,
              :(s += A_coeffs[ptr + $pi32] *
                     x[$(_x_idx(DIM, N, :(j - $Hi32 + $pi32)))...]))
    end

    # ----- Head: variable-length sum from input index 1 up to j + HWIDTH.
    # The pointer formula sums the lengths of all preceding head rows:
    #     ptr(j) = 1 + HWIDTH*(j-1) + (j-1)*j÷2.
    # `len` and `ptr_head` are computed once per thread; the inner for-loop
    # uses a runtime trip count, but it is bounded by WIDTH + HWIDTH and only
    # ever runs for the first `WIDTH` outputs along DIM.
    head_block = quote
        ptr_head = 1i32 + $Hi32 * (j - 1i32) + (j - 1i32) * j ÷ 2i32
        len      = j + $Hi32
        s        = zero(eltype(y))
        for k in 0i32:(len - 1i32)
            s += A_coeffs[ptr_head + k] *
                 x[$(_x_idx(DIM, N, :(1i32 + k)))...]
        end
    end

    # ----- Tail: mirror of the head, shrinking towards index M.
    # `jt = j - (M - WIDTH)` indexes into the tail region (1-based). After
    # all head and body coefficients the offset is (M-WIDTH)*WIDTH + 1, and
    # each tail row shrinks by one coefficient compared with the previous one.
    tail_block = quote
        jt       = j - (M - $Wi32)
        ptr_tail = (M - $Wi32) * $Wi32 + 1i32 +
                   ($Wi32 + $Hi32 + 1i32) * (jt - 1i32) -
                   (jt - 1i32) * jt ÷ 2i32
        len      = M - j + $Hi32 + 1i32
        s        = zero(eltype(y))
        for k in 0i32:(len - 1i32)
            s += A_coeffs[ptr_tail + k] *
                 x[$(_x_idx(DIM, N, :(j - $Hi32 + k)))...]
        end
    end

    return quote
        idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x - 1i32
        idx ≥ prod(sz) && return nothing

        $decomp

        j = $iDIM
        M = sz[$DIM]

        # `local s` declares the accumulator at function scope so the
        # assignments and `+=` updates inside the branches and the inner
        # for-loops update a single shared variable. Without this, Julia's
        # hard-scope rules would create a fresh `s` inside the for-loop and
        # discard the result.
        local s

        @inbounds if j ≤ $Wi32
            $head_block
        elseif j ≤ M - $Wi32
            $body_block
        else
            $tail_block
        end

        @inbounds y[$out_idx...] = s
        return nothing
    end
end


# ================================================================================
# Launch helpers
# ================================================================================

"""
    _check_shapes(y, A, x, ::Val{DIM})

Validate the array shape conventions shared by both GPU `mul!` methods.

The differentiated dimension of `x` and `y` must equal `size(A, 1)`, and the
two arrays must otherwise have identical shape. Throws `ArgumentError` with a
descriptive message if either condition is violated.
"""
@inline function _check_shapes(y, A, x, ::Val{DIM}) where {DIM}
    size(x, DIM) == size(y, DIM) == size(A, 1) ||
        throw(ArgumentError("inconsistent sizes"))
    size(x) == size(y) ||
        throw(ArgumentError("y and x must have the same shape"))
    return nothing
end

"""
    _launch_config(kernel, total, nthreads) -> (threads, blocks)

Decide how many threads-per-block and how many blocks to use for a kernel.

When `nthreads` is `nothing`, the per-block thread count is taken from
`launch_configuration` (occupancy heuristic) and capped at `total` for very
small launches. When `nthreads` is supplied, the caller's choice is honored
verbatim — useful for benchmarks pinning a block size.

`blocks` is computed by ceiling division of `total` by the chosen thread
count, so the launch covers all `total` outputs while leaving a possibly
under-filled final block.
"""
@inline function _launch_config(kernel, total::Int, nthreads)
    config  = launch_configuration(kernel.fun)
    threads = isnothing(nthreads) ? min(config.threads, total) : nthreads
    blocks  = cld(total, threads)
    return threads, blocks
end

"""
    optimal_forward_threads(y, A::DiffMatrix, x, ::Val{DIM}; max_threads=nothing)

Determine the optimal number of threads required to run the
[`_gpu_forward_kernel`](@ref).

Run this function and store the results for later use when calling [`mul!`](@ref)
with GPU data.
"""
function FDGrids.optimal_forward_threads(y::AbstractArray,
                                         A::DiffMatrix{T, WIDTH},
                                         x::AbstractArray,
                                          ::Val{DIM};
                               max_threads=nothing) where {T, WIDTH, DIM}
    k = @cuda launch=false _gpu_forward_kernel!(
        y, A.coeffs, x, Int32.(size(x)), Val(Int32(DIM)), Val(Int32(WIDTH))
    )
    threads = launch_configuration(k.fun).threads
    return isnothing(max_threads) ? threads : min(threads, max_threads)
end

"""
    optimal_adjoint_threads(y, A::DiffMatrix, x, ::Val{DIM}; max_threads=nothing)

Determine the optimal number of threads required to run the
[`_gpu_adjoint_kernel`](@ref).

Run this function and store the results for later use when calling [`mul!`](@ref)
with GPU data.
"""
function FDGrids.optimal_adjoint_threads(y::AbstractArray,
                                         A::AdjointDiffMatrix{T, WIDTH},
                                         x::AbstractArray,
                                          ::Val{DIM};
                               max_threads=nothing) where {T, WIDTH, DIM}
    k = @cuda launch=false _gpu_adjoint_kernel!(
        y, A.coeffs, x, Int32.(size(x)), Val(Int32(DIM)), Val(Int32(WIDTH))
    )
    threads = launch_configuration(k.fun).threads
    return isnothing(max_threads) ? threads : min(threads, max_threads)
end


# ================================================================================
# Public mul! dispatch
# ================================================================================

"""
    LinearAlgebra.mul!(y, A::DiffMatrix{T, WIDTH, OPTIMISE, <:CuArray},
                       x, ::Val{DIM} = Val(1); nthreads = nothing) -> y

Apply the forward finite-difference operator `A` on the GPU.

The method dispatches on the coefficient backing of `A` being a `CuArray`, so
the CPU `mul!` is unaffected. `y` and `x` may be `CuArray`s or any
GPU-resident wrapper that satisfies the abstract array interface — the kernel
goes through `getindex`/`setindex!` and is therefore agnostic to the concrete
device array type.

The host validates shapes, picks a launch configuration via
`launch_configuration`, and dispatches one thread per element of `y`. The
`nthreads` keyword overrides the per-block thread count and is intended for
benchmarks; the auto-tuned default is sufficient for production use.

# Arguments
- `y`: output array, same shape as `x`.
- `A`: a GPU-resident `DiffMatrix`.
- `x`: input array; `size(x, DIM)` must equal `size(A, 1)`.
- `Val{DIM}`: dimension to differentiate along, defaults to `1`.
- `nthreads`: optional per-block thread-count override.

# Examples
```julia
using CUDA, FDGrids, LinearAlgebra

xs = range(-1, 1; length = 256)
D  = DiffMatrix(xs, 5, 1)
Dg = cu(D)

u  = CuArray(Float32.(sin.(xs)))
du = similar(u)
mul!(du, Dg, u)
```
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::DiffMatrix{T, WIDTH, OPTIMISE, <:CuArray},
                            x::AbstractArray{S, N},
                             ::Val{DIM} =Val(1);
                            nthreads::TH=nothing
                            ) where {T, S, N, WIDTH, OPTIMISE, DIM, TH<:Union{Nothing, Int}}
    # Rank cap of 4 reflects the rank cap of the CPU code path and keeps the
    # generated kernel compilation footprint bounded. Most callers use N ≤ 3.
    N   in 1:4 || throw(ArgumentError("N must be in 1:4"))
    DIM in 1:N || throw(ArgumentError("DIM must be in 1:N"))
    _check_shapes(y, A, x, Val(DIM))

    # `sz` is passed as a kernel argument so the device kernel can decompose
    # the flat thread id into N-D indices without a dynamic shape query.
    sz     = Int32.(size(x))
    total  = length(x)

    if TH <: Nothing
        kernel = @cuda launch=false _gpu_forward_kernel!(
            y, A.coeffs, x, sz, Val(Int32(DIM)), Val(Int32(WIDTH)))

        threads, blocks = _launch_config(kernel, total, nthreads)
        kernel(y, A.coeffs, x, sz, Val(Int32(DIM)), Val(Int32(WIDTH));
            threads, blocks)
    else
        @cuda threads=nthreads blocks=cld(total, nthreads) _gpu_forward_kernel!(
            y, A.coeffs, x, sz, Val(Int32(DIM)), Val(Int32(WIDTH))
        )
    end

    return y
end

"""
    LinearAlgebra.mul!(y, A::AdjointDiffMatrix{T, WIDTH, P, <:CuArray},
                       x, ::Val{DIM} = Val(1); nthreads = nothing) -> y

Apply the transposed finite-difference operator `A` on the GPU.

This method shares the launch boilerplate with the forward GPU `mul!`. The
parent matrix type `P` is left free, so weighted and unweighted adjoints
dispatch to the same kernel — only `A.coeffs` (the precomputed adjoint
coefficient vector) is read on the device, and any weighting was already
folded into those coefficients at construction time.

The adjoint requires `size(A, 1) > 2 * WIDTH`, which is the same constraint
the CPU adjoint constructor enforces; it is re-checked here as a defence in
depth in case the operator was constructed by hand.

# Arguments
- `y`: output array, same shape as `x`.
- `A`: a GPU-resident `AdjointDiffMatrix`.
- `x`: input array; `size(x, DIM)` must equal `size(A, 1)`.
- `Val{DIM}`: dimension to apply the adjoint along, defaults to `1`.
- `nthreads`: optional per-block thread-count override.

# Examples
```julia
using CUDA, FDGrids, LinearAlgebra

g  = grid(256, -1, 1, GaussLobattoGrid())
D  = DiffMatrix(g.xs, 5, 1)
A  = adjoint(D, g.ws)            # weighted adjoint
Ag = cu(A)

v  = CuArray(Float32.(randn(length(g.xs))))
y  = similar(v)
mul!(y, Ag, v)
```
"""
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            A::AdjointDiffMatrix{T, WIDTH, P, <:CuArray},
                            x::AbstractArray{S, N},
                             ::Val{DIM} =Val(1);
                            nthreads::TH=nothing
                            ) where {T, S, N, WIDTH, P, DIM, TH<:Union{Nothing, Int}}
    N   in 1:4 || throw(ArgumentError("N must be in 1:4"))
    DIM in 1:N || throw(ArgumentError("DIM must be in 1:N"))
    size(A, 1) > 2 * WIDTH ||
        throw(ArgumentError("GPU adjoint requires size(A,1) > 2*WIDTH"))
    _check_shapes(y, A, x, Val(DIM))

    sz     = Int32.(size(x))
    total  = length(x)

    if TH <: Nothing
        kernel = @cuda launch=false _gpu_adjoint_kernel!(
            y, A.coeffs, x, sz, Val(Int32(DIM)), Val(Int32(WIDTH)))

        threads, blocks = _launch_config(kernel, total, nthreads)
        kernel(y, A.coeffs, x, sz, Val(Int32(DIM)), Val(Int32(WIDTH));
            threads, blocks)
    else
        @cuda threads=nthreads blocks=cld(total, nthreads) _gpu_adjoint_kernel!(
            y, A.coeffs, x, sz, Val(Int32(DIM)), Val(Int32(WIDTH))
        )
    end

    return y
end

end
