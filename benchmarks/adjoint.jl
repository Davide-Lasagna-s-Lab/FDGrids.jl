# Benchmark: forward DiffMatrix mul! vs adjoint mul! throughput.
#
# For each (ndim, width, dim, sz) configuration the benchmark measures both
#   mul!(y, D,         x, Val(dim))   — forward operator
#   mul!(y, adjoint(D), x, Val(dim))  — adjoint operator
#
# All configurations are run in randomised order over NRUNS independent passes
# to spread thermal and cache-state effects.  The median over all passes is used
# for the final plot.
#
# Each panel shows the speedup ratio (forward time / adjoint time) on a log₂
# y-axis with a parity line at 1.  Colour encodes stencil width; line style
# encodes the differentiation direction.
#
# Saves adjoint.svg to ../docs/src/assets/benchmarks/.
#
# Setup (once):
#   julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#   julia --project=benchmarks benchmarks/adjoint.jl

using BenchmarkTools, LinearAlgebra, FDGrids, CairoMakie, Printf, Random, Statistics

const SIZES_BY_NDIM = Dict(
    1 => [32, 64, 128, 256, 512, 1024, 2048, 4096],
    2 => [16, 32, 64, 128, 256, 512, 1024],
    3 => [8,  16, 32, 64,  128],
    4 => [8,  12, 16, 24,  32],
)
const WIDTHS = [3, 5, 7]
const ORDER  = 1
const NRUNS  = 10

function time_fwd_adj(sz, ndim, dim, width)
    sizes = ntuple(_ -> sz, ndim)
    xs    = collect(range(0.0, 1.0; length=sz))
    D     = DiffMatrix(xs, width, ORDER)
    Dt    = adjoint(D)
    x     = randn(sizes...)
    y     = similar(x)
    t_fwd = @belapsed mul!($y, $D,  $x, $(Val(dim))) evals=5 samples=3
    t_adj = @belapsed mul!($y, $Dt, $x, $(Val(dim))) evals=5 samples=3
    return t_fwd, t_adj
end

# ─── Collect timings ─────────────────────────────────────────────────────────
println("Building configuration list…")

configs = NamedTuple[]
for ndim in 1:4, sz in SIZES_BY_NDIM[ndim], width in WIDTHS, dim in 1:ndim
    sz > 2 * width && push!(configs, (ndim=ndim, width=width, dim=dim, sz=sz))
end
println("  $(length(configs)) configurations × $NRUNS runs = $(length(configs)*NRUNS) calls\n")

timings_fwd = Dict{NTuple{4,Int}, Vector{Float64}}(
    (c.ndim, c.width, c.dim, c.sz) => Float64[] for c in configs)
timings_adj = Dict{NTuple{4,Int}, Vector{Float64}}(
    (c.ndim, c.width, c.dim, c.sz) => Float64[] for c in configs)

for run in 1:NRUNS
    println("Pass $run / $NRUNS …")
    shuffle!(configs)
    for cfg in configs
        tf, ta  = time_fwd_adj(cfg.sz, cfg.ndim, cfg.dim, cfg.width)
        key     = (cfg.ndim, cfg.width, cfg.dim, cfg.sz)
        push!(timings_fwd[key], tf)
        push!(timings_adj[key], ta)
        total = cfg.sz ^ cfg.ndim
        @printf "  %dD  sz=%4d  width=%d  dim=%d  fwd %6.2f μs  adj %6.2f μs  (%.2f / %.2f GEl/s)\n" cfg.ndim cfg.sz cfg.width cfg.dim tf*1e6 ta*1e6 total/tf/1e9 total/ta/1e9
    end
end

# ─── Figure ──────────────────────────────────────────────────────────────────
colors    = Makie.wong_colors()
width_col = Dict(3 => colors[1], 5 => colors[2], 7 => colors[3])
dim_style = [:solid, :dash, :dashdot, :dot]
sups      = ["", "²", "³", "⁴"]

fig = Figure(size=(1100, 900))
Label(fig[0, :],
    "Adjoint speedup over forward mul! — forward time / adjoint time" *
    "\nValues above 1: adjoint is faster  ·  Values below 1: adjoint has overhead";
    fontsize=12, font=:bold)

for ndim in 1:4
    row       = cld(ndim, 2)
    col       = isodd(ndim) ? 1 : 2
    all_sizes = SIZES_BY_NDIM[ndim]
    sup       = sups[ndim]

    ax = Axis(fig[row, col];
        title   = "$(ndim)D array",
        xlabel  = "Array size",
        ylabel  = "Speedup (forward time / adjoint time)",
        xscale  = log2,
        xticks  = (Float64.(all_sizes), ["$sz$sup" for sz in all_sizes]),
        xticklabelrotation = π/4)

    hlines!(ax, [1.0]; linestyle=:dash, color=:black, linewidth=1)

    for width in WIDTHS
        valid_sizes = [sz for sz in all_sizes if sz > 2 * width]
        xs_plot     = Float64.(valid_sizes)
        for dim in 1:ndim
            tf      = [median(timings_fwd[(ndim, width, dim, sz)]) for sz in valid_sizes]
            ta      = [median(timings_adj[(ndim, width, dim, sz)]) for sz in valid_sizes]
            speedup = tf ./ ta
            lines!(ax, xs_plot, speedup;
                color=width_col[width], linestyle=dim_style[dim], linewidth=1.5)
            scatter!(ax, xs_plot, speedup;
                color=width_col[width], markersize=5)
        end
    end
end

# Legends
width_elems = [LineElement(color=width_col[w], linewidth=2) for w in WIDTHS]
dim_elems   = [LineElement(color=:black, linestyle=dim_style[d], linewidth=2) for d in 1:4]

Legend(fig[3, 1], width_elems, ["width $w" for w in WIDTHS];
    orientation=:horizontal, framevisible=false, tellwidth=false, title="Stencil width")
Legend(fig[3, 2], dim_elems, ["dim $d" for d in 1:4];
    orientation=:horizontal, framevisible=false, tellwidth=false, title="Differentiation direction")

cpu    = Sys.cpu_info()[1].model
mem_gb = round(Int, Sys.total_memory() / 2^30)
Label(fig[4, :], "CPU: $cpu  |  RAM: $(mem_gb) GB"; fontsize=9, color=:gray50)

outdir  = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(outdir)
outfile = joinpath(outdir, "adjoint.svg")
save(outfile, fig)
println("\nSaved → $outfile")
