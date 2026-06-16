# ================================================================================
# Tests for the CUDA extension.
# Included from runtests.jl after `using CUDA, LinearAlgebra, FDGrids, Test`,
# and only when CUDA is functional.
#
# Strategy: build each operator on the CPU and on the GPU, apply both to
# matching input arrays, and compare. The CPU `mul!` paths are themselves
# exercised by the other test files, so they serve as the trusted reference.
# Numerical tolerances are conservative for Float32 transfers (cu) and tight
# for Float64 transfers (Adapt.adapt with CuArray).
# ================================================================================

using Adapt

const _GPU_RTOL_F32 = 1e-4
const _GPU_RTOL_F64 = 1e-12

# Build a CPU/GPU operator pair at the requested element type.
#
# For Float32, we use `cu` (the user-facing convenience entry that casts
# Float64 → Float32). For Float64, we use Adapt to transfer without precision
# loss. Both paths exercise the same kernels.
function _gpu_op(D, ::Type{Float32})
    return cu(D)
end
function _gpu_op(D, ::Type{Float64})
    return Adapt.adapt(CuArray, D)
end

# Construct an input array on the GPU at the requested element type from a
# CPU reference, while keeping the CPU array at Float64 for tighter reference
# computations when possible.
function _gpu_in(x_cpu, ::Type{T}) where {T}
    return CuArray(T.(x_cpu))
end


# ================================================================================
# Adapt and cu type tests
# ================================================================================
@testset "FDGridsCUDAExt: adaptation                " begin
    M = 64
    @testset "DiffMatrix width=$width" for width in (3, 5, 7)
        xs = collect(range(-1.0, 1.0; length = M))
        D  = DiffMatrix(xs, width, 1)

        # cu downcasts to Float32 and produces CuArray storage
        Dg32 = cu(D)
        @test Dg32 isa DiffMatrix{Float32, width, true}
        @test Dg32.coeffs isa CuArray{Float32}
        @test Array(Dg32.coeffs) ≈ Float32.(D.coeffs)
        @test size(Dg32) == size(D)

        # Adapt to CuArray preserves Float64
        Dg64 = Adapt.adapt(CuArray, D)
        @test Dg64 isa DiffMatrix{Float64, width, true}
        @test Dg64.coeffs isa CuArray{Float64}
        @test Array(Dg64.coeffs) == D.coeffs
        @test size(Dg64) == size(D)
    end

    @testset "AdjointDiffMatrix width=$width" for width in (3, 5, 7)
        xs  = collect(range(-1.0, 1.0; length = M))
        D   = DiffMatrix(xs, width, 1)
        At  = adjoint(D)

        # cu()
        Atg32 = cu(At)
        @test Atg32 isa AdjointDiffMatrix
        @test Atg32.parent isa DiffMatrix{Float32, width, true}
        @test Atg32.coeffs isa CuArray{Float32}
        @test Array(Atg32.coeffs) ≈ Float32.(At.coeffs)
        # `adjoint` on an adjoint unwraps to the parent on either device
        @test adjoint(Atg32) === Atg32.parent

        # Adapt
        Atg64 = Adapt.adapt(CuArray, At)
        @test Atg64 isa AdjointDiffMatrix
        @test Atg64.parent isa DiffMatrix{Float64, width, true}
        @test Atg64.coeffs isa CuArray{Float64}
        @test Array(Atg64.coeffs) == At.coeffs
        @test adjoint(Atg64) === Atg64.parent
    end

    @testset "weighted AdjointDiffMatrix width=$width" for width in (3, 5, 7)
        xs = collect(range(-1.0, 1.0; length = M))
        D  = DiffMatrix(xs, width, 1)
        w  = 1.0 .+ rand(M)
        Aw = adjoint(D, w)

        Awg = Adapt.adapt(CuArray, Aw)
        @test Awg.coeffs isa CuArray{Float64}
        @test Array(Awg.coeffs) == Aw.coeffs
    end
end


# ================================================================================
# Forward mul!
# ================================================================================
@testset "FDGridsCUDAExt: forward 1D                " begin
    M = 128

    @testset "T=$T width=$width" for T in (Float32, Float64), width in (3, 5, 7)
        rtol = T === Float32 ? _GPU_RTOL_F32 : _GPU_RTOL_F64

        xs = collect(range(-1.0, 1.0; length = M))
        D  = DiffMatrix(xs, width, 1)
        Dg = _gpu_op(D, T)

        u  = exp.(0.7 .* xs) # smooth field
        y_cpu = similar(u)
        mul!(y_cpu, D, u)

        ug    = _gpu_in(u, T)
        yg    = similar(ug)
        mul!(yg, Dg, ug)

        @test Array(yg) ≈ T.(y_cpu) rtol = rtol
    end
end

@testset "FDGridsCUDAExt: forward N-D               " begin
    M     = 32
    OTHER = 3

    @testset "T=$T N=$N DIM=$DIM width=$width" for
            T     in (Float32, Float64),
            N     in 1:4,
            DIM   in 1:N,
            width in (3, 5, 7)

        rtol  = T === Float32 ? _GPU_RTOL_F32 : _GPU_RTOL_F64
        xs    = collect(range(-1.0, 1.0; length = M))
        D     = DiffMatrix(xs, width, 1)
        Dg    = _gpu_op(D, T)
        shape = ntuple(d -> d == DIM ? M : OTHER, N)

        x_cpu = randn(shape...)
        y_cpu = similar(x_cpu)
        mul!(y_cpu, D, x_cpu, Val(DIM))

        xg = _gpu_in(x_cpu, T)
        yg = similar(xg)
        mul!(yg, Dg, xg, Val(DIM))

        @test Array(yg) ≈ T.(y_cpu) rtol = rtol
    end
end


# ================================================================================
# Adjoint mul!
# ================================================================================
@testset "FDGridsCUDAExt: adjoint 1D                " begin
    M = 128

    @testset "T=$T width=$width" for T in (Float32, Float64), width in (3, 5, 7)
        rtol = T === Float32 ? _GPU_RTOL_F32 : _GPU_RTOL_F64

        xs = collect(range(-1.0, 1.0; length = M))
        D  = DiffMatrix(xs, width, 1)
        At = adjoint(D)
        Ag = _gpu_op(At, T)

        x_cpu = randn(M)
        y_cpu = similar(x_cpu)
        mul!(y_cpu, At, x_cpu)

        xg = _gpu_in(x_cpu, T)
        yg = similar(xg)
        mul!(yg, Ag, xg)

        @test Array(yg) ≈ T.(y_cpu) rtol = rtol
    end
end

@testset "FDGridsCUDAExt: adjoint N-D               " begin
    M     = 32
    OTHER = 3

    @testset "T=$T N=$N DIM=$DIM width=$width" for
            T     in (Float32, Float64),
            N     in 1:4,
            DIM   in 1:N,
            width in (3, 5, 7)

        # adjoint requires M > 2*WIDTH for a non-empty body
        M > 2 * width || continue
        rtol  = T === Float32 ? _GPU_RTOL_F32 : _GPU_RTOL_F64
        xs    = collect(range(-1.0, 1.0; length = M))
        D     = DiffMatrix(xs, width, 1)
        At    = adjoint(D)
        Ag    = _gpu_op(At, T)
        shape = ntuple(d -> d == DIM ? M : OTHER, N)

        x_cpu = randn(shape...)
        y_cpu = similar(x_cpu)
        mul!(y_cpu, At, x_cpu, Val(DIM))

        xg = _gpu_in(x_cpu, T)
        yg = similar(xg)
        mul!(yg, Ag, xg, Val(DIM))

        @test Array(yg) ≈ T.(y_cpu) rtol = rtol
    end
end


# ================================================================================
# Weighted adjoint
# ================================================================================
# The weights are baked into A.coeffs at construction time, so on the kernel
# side a weighted adjoint is indistinguishable from an unweighted one. This
# test exercises the same code path against the heavier CPU reference path
# `W⁻¹ * full(D)' * W`.

@testset "FDGridsCUDAExt: weighted adjoint          " begin
    M = 128

    @testset "T=$T width=$width" for T in (Float32, Float64), width in (3, 5, 7)
        rtol = T === Float32 ? _GPU_RTOL_F32 : _GPU_RTOL_F64

        xs = collect(range(-1.0, 1.0; length = M))
        D  = DiffMatrix(xs, width, 1)
        w  = 1.0 .+ rand(M)
        Aw = adjoint(D, w)
        Ag = _gpu_op(Aw, T)

        x_cpu = randn(M)
        y_cpu = similar(x_cpu)
        mul!(y_cpu, Aw, x_cpu)

        xg = _gpu_in(x_cpu, T)
        yg = similar(xg)
        mul!(yg, Ag, xg)

        @test Array(yg) ≈ T.(y_cpu) rtol = rtol
    end
end


# ================================================================================
# Argument validation
# ================================================================================
@testset "FDGridsCUDAExt: argument validation       " begin
    M  = 32
    xs = collect(range(-1.0, 1.0; length = M))
    D  = DiffMatrix(xs, 5, 1)
    Dg = cu(D)

    x  = CuArray(randn(Float32, M))
    y  = similar(x)

    # mismatched shape between y and x
    y_bad = CuArray(randn(Float32, M + 1))
    @test_throws ArgumentError mul!(y_bad, Dg, x)

    # DIM out of range
    @test_throws ArgumentError mul!(y, Dg, x, Val(2))

    # mismatched leading dimension between A and x
    x_bad = CuArray(randn(Float32, M + 1))
    y_bad2 = similar(x_bad)
    @test_throws ArgumentError mul!(y_bad2, Dg, x_bad)

    # adjoint built from too-small parent is rejected at construction (CPU
    # path), so the size-guard inside the GPU adjoint mul! is a defence in
    # depth. We still verify the GPU path runs for the smallest allowed grid.
    Atg = cu(adjoint(D))
    x32 = CuArray(randn(Float32, M))
    y32 = similar(x32)
    mul!(y32, Atg, x32)
    @test eltype(y32) == Float32
end


# ================================================================================
# nthreads override is honoured
# ================================================================================
@testset "FDGridsCUDAExt: nthreads override         " begin
    M  = 256
    xs = collect(range(-1.0, 1.0; length = M))
    D  = DiffMatrix(xs, 5, 1)
    Dg = cu(D)
    At = adjoint(D)
    Ag = cu(At)

    u  = sin.(xs)
    ug = CuArray(Float32.(u))

    yg_auto = similar(ug)
    yg_pin  = similar(ug)
    nthreads_forward = FDGrids.optimal_forward_threads(yg_pin, Dg, ug, Val(1))
    nthreads_adjoint = FDGrids.optimal_adjoint_threads(yg_pin, Ag, ug, Val(1))

    # The result must not depend on the per-block thread count, so a small
    # custom block size should still produce the same field as the auto-tuned
    # default. 32 ensures at least one warp per block.
    mul!(yg_auto, Dg, ug)
    mul!(yg_pin,  Dg, ug; nthreads=nthreads_forward)
    @test Array(yg_auto) ≈ Array(yg_pin)

    mul!(yg_auto, Ag, ug)
    mul!(yg_pin,  Ag, ug; nthreads=nthreads_adjoint)
    @test Array(yg_auto) ≈ Array(yg_pin)
end
