# Benchmark: GPU speedup over CPU for forward and adjoint `mul!`.
#
# For each (ndim, width, dim, sz) configuration, the script times the same
# `mul!` call on the CPU and on the GPU, then plots the speedup
# (CPU time / GPU time) on a log₂ y-axis. A horizontal parity line at
# 1 separates "GPU faster" from "CPU faster".
#
# Two figures are produced, mirroring the layout used by matmul.jl and
# adjoint.jl:
#   gpu_speedup_forward.svg — forward `mul!` speedup, 4 panels (ndim 1–4)
#   gpu_speedup_adjoint.svg — adjoint `mul!` speedup, 4 panels (ndim 1–4)
# Within each panel, colour encodes stencil width and linestyle encodes the
# differentiated dimension.
#
# Both sides run in Float32 for a fair comparison: `cu` downcasts to Float32
# by default, so building the CPU reference at Float32 keeps the numerical
# work identical.
#
# GPU timing uses `CUDA.@sync` so kernel completion is measured rather than
# just the launch overhead. Each configuration is warmed up once before the
# first measurement so kernel compilation is not folded into the timings.
#
# Configurations run in randomised order over NRUNS independent passes. This
# spreads thermal effects and host/device contention across the grid rather
# than concentrating them at the start.
#
# Setup (once):
#   julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run (requires a CUDA-capable device):
#   julia --project=benchmarks benchmarks/gpu.jl

using BenchmarkTools, LinearAlgebra, FDGrids, CUDA, CairoMakie, Printf, Random, Statistics

# ----- Configuration ---------------------------------------------------------
# Sizes per dimension are chosen so the largest array in each panel reaches
# tens of millions of elements. That is well within the per-launch element cap
# (~2.1G with Int32 indexing) and well into the regime where GPU parallelism
# dominates over launch overhead.
const SIZES_BY_NDIM = Dict(
    1 => [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 262144],
    2 => [16,  32,  64,  128,  256,  512,  1024,  2048],
    3 => [8,   16,  32,  64,   128,  256],
    4 => [8,   12,  16,  24,   32,   48,   64],
)
const WIDTHS = [3, 5, 7]
const ORDER  = 1
const NRUNS  = 5     # GPU runs are slower per call, so fewer passes.
const ELTYPE = Float32

# ----- Timing ----------------------------------------------------------------
# Float32 throughout: cu() downcasts Float64 → Float32 by default, so we build
# the CPU reference at Float32 to compare equivalent numerical work.

function time_cpu(A, x, y, dim)
    return @belapsed mul!($y, $A, $x, $(Val(dim))) evals=5 samples=3
end

function time_gpu(Ag, xg, yg, dim)
    # CUDA.@sync waits for the kernel to finish before @belapsed stops the
    # clock. Without it, @belapsed would measure only the host-side launch
    # overhead.
    return @belapsed CUDA.@sync mul!($yg, $Ag, $xg, $(Val(dim))) evals=5 samples=3
end

# Build operators and arrays for a single configuration. We keep both the CPU
# and GPU operators in the returned tuple so the timing loop can call each
# timer repeatedly without rebuilding the arrays.
function setup(sz, ndim, dim, width)
    sz > 2 * width || return nothing   # adjoint requires this

    sizes = ntuple(_ -> sz, ndim)
    xs    = collect(range(0f0, 1f0; length = sz))

    D     = DiffMatrix(xs, width, ORDER, ELTYPE)
    At    = adjoint(D)

    Dg    = cu(D)
    Atg   = cu(At)

    x     = randn(ELTYPE, sizes...)
    y     = similar(x)
    xg    = CuArray(x)
    yg    = similar(xg)

    # Warm up: triggers CUDA kernel compilation outside the timed region.
    CUDA.@sync mul!(yg, Dg,  xg, Val(dim))
    CUDA.@sync mul!(yg, Atg, xg, Val(dim))

    return (; D, At, Dg, Atg, x, y, xg, yg)
end

# ----- Collect timings -------------------------------------------------------
println("Building configuration list…")

configs = NamedTuple[]
for ndim in 1:4, sz in SIZES_BY_NDIM[ndim], width in WIDTHS, dim in 1:ndim
    sz > 2 * width && push!(configs,
        (ndim = ndim, width = width, dim = dim, sz = sz))
end
println("  $(length(configs)) configurations × $NRUNS runs = $(length(configs) * NRUNS) calls\n")

# Each map holds the per-configuration vector of independent timings.
const Key = NTuple{4, Int}   # (ndim, width, dim, sz)
make_dict() = Dict{Key, Vector{Float64}}(
    (c.ndim, c.width, c.dim, c.sz) => Float64[] for c in configs)

timings_cpu_fwd = make_dict()
timings_gpu_fwd = make_dict()
timings_cpu_adj = make_dict()
timings_gpu_adj = make_dict()

# Show what we are running on.
println("Device : ", CUDA.name(CUDA.device()))
println("Driver : ", CUDA.driver_version())
println("Runtime: ", CUDA.runtime_version())
println()

for run in 1:NRUNS
    println("Pass $run / $NRUNS …")
    shuffle!(configs)
    for cfg in configs
        ctx = setup(cfg.sz, cfg.ndim, cfg.dim, cfg.width)
        ctx === nothing && continue

        tcf = time_cpu(ctx.D,   ctx.x,  ctx.y,  cfg.dim)
        tgf = time_gpu(ctx.Dg,  ctx.xg, ctx.yg, cfg.dim)
        tca = time_cpu(ctx.At,  ctx.x,  ctx.y,  cfg.dim)
        tga = time_gpu(ctx.Atg, ctx.xg, ctx.yg, cfg.dim)

        key = (cfg.ndim, cfg.width, cfg.dim, cfg.sz)
        push!(timings_cpu_fwd[key], tcf)
        push!(timings_gpu_fwd[key], tgf)
        push!(timings_cpu_adj[key], tca)
        push!(timings_gpu_adj[key], tga)

        @printf("  %dD  sz=%-6d width=%d dim=%d   fwd: CPU %7.1f μs  GPU %7.1f μs  (×%5.1f)   adj: CPU %7.1f μs  GPU %7.1f μs  (×%5.1f)\n",
                cfg.ndim, cfg.sz, cfg.width, cfg.dim,
                tcf * 1e6, tgf * 1e6, tcf / tgf,
                tca * 1e6, tga * 1e6, tca / tga)
    end
end

# ----- Figure helper ---------------------------------------------------------
# Both figures use the same layout: 4 panels (one per ndim), with stencil
# width as colour and differentiation direction as line style. This mirrors
# matmul.jl and adjoint.jl so the GPU plots can be read alongside them.

colors    = Makie.wong_colors()
width_col = Dict(3 => colors[1], 5 => colors[2], 7 => colors[3])
dim_style = [:solid, :dash, :dashdot, :dot]
sups      = ["", "²", "³", "⁴"]

cpu_name = Sys.cpu_info()[1].model
gpu_name = CUDA.name(CUDA.device())
mem_gb   = round(Int, Sys.total_memory() / 2^30)

function build_figure(title_text, timings_cpu, timings_gpu, outfile)
    fig = Figure(size = (1100, 900))
    Label(fig[0, :],
        title_text * "\n" *
        "Speedup = CPU time / GPU time   ·   parity line at 1   ·   Float32";
        fontsize = 13, font = :bold)

    for ndim in 1:4
        row       = cld(ndim, 2)
        col       = isodd(ndim) ? 1 : 2
        all_sizes = SIZES_BY_NDIM[ndim]
        sup       = sups[ndim]

        ax = Axis(fig[row, col];
            title   = "$(ndim)D array",
            xlabel  = "Array size",
            ylabel  = "Speedup (CPU / GPU)",
            xscale  = log2,
            yscale  = log2,
            xticks  = (Float64.(all_sizes), ["$sz$sup" for sz in all_sizes]),
            xticklabelrotation = π/4,
            yticks  = [2.0^i for i in -4:10],
            ytickformat = ys -> [@sprintf("%g", y) for y in ys])

        # Parity reference: above this, GPU wins; below, CPU wins.
        hlines!(ax, [1.0]; linestyle = :dash, color = :black, linewidth = 1)

        for width in WIDTHS
            valid_sizes = [sz for sz in all_sizes if sz > 2 * width]
            xs_plot     = Float64.(valid_sizes)
            for dim in 1:ndim
                spd = [median(timings_cpu[(ndim, width, dim, sz)]) /
                       median(timings_gpu[(ndim, width, dim, sz)])
                       for sz in valid_sizes]
                lines!(ax, xs_plot, spd;
                    color = width_col[width],
                    linestyle = dim_style[dim],
                    linewidth = 1.5)
                scatter!(ax, xs_plot, spd;
                    color = width_col[width], markersize = 5)
            end
        end
    end

    # Legends: colour = stencil width, linestyle = differentiation direction.
    width_elems = [LineElement(color = width_col[w], linewidth = 2) for w in WIDTHS]
    dim_elems   = [LineElement(color = :black, linestyle = dim_style[d], linewidth = 2) for d in 1:4]

    Legend(fig[3, 1], width_elems, ["width $w" for w in WIDTHS];
        orientation = :horizontal, framevisible = false, tellwidth = false,
        title = "Stencil width")
    Legend(fig[3, 2], dim_elems, ["dim $d" for d in 1:4];
        orientation = :horizontal, framevisible = false, tellwidth = false,
        title = "Differentiation direction")

    Label(fig[4, :],
        "CPU: $cpu_name  |  GPU: $gpu_name  |  RAM: $(mem_gb) GB";
        fontsize = 9, color = :gray50)

    save(outfile, fig)
    println("Saved → $outfile")
end

outdir = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(outdir)

println()
build_figure(
    "GPU speedup over CPU — forward `mul!` along each array axis",
    timings_cpu_fwd, timings_gpu_fwd,
    joinpath(outdir, "gpu_speedup_forward.svg"))

build_figure(
    "GPU speedup over CPU — adjoint `mul!` along each array axis",
    timings_cpu_adj, timings_gpu_adj,
    joinpath(outdir, "gpu_speedup_adjoint.svg"))
