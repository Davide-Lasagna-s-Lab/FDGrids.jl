# Benchmark: DiffMatrix mul! across array dimensions 1–4, stencil widths 3/5/7,
# and all differentiation directions.
#
# All (ndim, width, dim, sz) combinations are collected into a list and run in
# randomised order, repeated NRUNS times.  This spreads thermal effects and
# cache-state bias across all configurations rather than concentrating them.
# The median over NRUNS independent measurements is used for the final plot.
#
# Saves matmul.svg to ../docs/src/assets/benchmarks/.
#
# Setup (once):
#   julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#   julia --project=benchmarks benchmarks/matmul.jl

using BenchmarkTools, LinearAlgebra, FDGrids, CairoMakie, Printf, Random, Statistics

const SIZES_BY_NDIM = Dict(
    1 => [32, 64, 128, 256, 512, 1024, 2048, 4096],
    2 => [16, 32, 64, 128, 256, 512, 1024],
    3 => [8,  16, 32, 64,  128],
    4 => [8,  12, 16, 24,  32],
)
const WIDTHS = [3, 5, 7]
const ORDER  = 1
const NRUNS  = 10   # independent randomised passes over all configurations

function time_mul(sz, ndim, dim, width)
    sizes = ntuple(_ -> sz, ndim)
    xs    = collect(range(0.0, 1.0; length=sz))
    D     = DiffMatrix(xs, width, ORDER)
    x     = randn(sizes...)
    y     = similar(x)
    return @belapsed mul!($y, $D, $x, $(Val(dim))) evals=5 samples=3
end

# ─── Collect timings ─────────────────────────────────────────────────────────
println("Building configuration list…")

configs = NamedTuple[]
for ndim in 1:4, sz in SIZES_BY_NDIM[ndim], width in WIDTHS, dim in 1:ndim
    sz >= width && push!(configs, (ndim=ndim, width=width, dim=dim, sz=sz))
end
println("  $(length(configs)) configurations × $NRUNS runs = $(length(configs)*NRUNS) calls\n")

# results[(ndim, width, dim, sz)] → timings collected across all passes
timings = Dict{NTuple{4,Int}, Vector{Float64}}(
    (cfg.ndim, cfg.width, cfg.dim, cfg.sz) => Float64[] for cfg in configs)

for run in 1:NRUNS
    println("Pass $run / $NRUNS …")
    shuffle!(configs)
    for cfg in configs
        t     = time_mul(cfg.sz, cfg.ndim, cfg.dim, cfg.width)
        total = cfg.sz ^ cfg.ndim
        push!(timings[(cfg.ndim, cfg.width, cfg.dim, cfg.sz)], t)
        @printf "  %dD  sz=%4d  width=%d  dim=%d  %6.2f μs  (%5.2f GEl/s)\n" cfg.ndim cfg.sz cfg.width cfg.dim t*1e6 total/t/1e9
    end
end

# ─── Figure ──────────────────────────────────────────────────────────────────
colors    = Makie.wong_colors()
width_col = Dict(3 => colors[1], 5 => colors[2], 7 => colors[3])
dim_style = [:solid, :dash, :dashdot, :dot]
sups      = ["", "²", "³", "⁴"]

fig = Figure(size=(1100, 900))
Label(fig[0, :],
    "mul! throughput — DiffMatrix applied along each array axis" *
    "\nGEl/s = 10⁹ array elements processed per second  (higher is better)";
    fontsize=13, font=:bold)

for ndim in 1:4
    row       = cld(ndim, 2)
    col       = isodd(ndim) ? 1 : 2
    all_sizes = SIZES_BY_NDIM[ndim]
    sup       = sups[ndim]

    ax = Axis(fig[row, col];
        title   = "$(ndim)D array",
        xlabel  = "Array size",
        ylabel  = "Throughput (GEl/s)",
        xscale  = log2,
        yscale  = log2,
        xticks  = (Float64.(all_sizes), ["$sz$sup" for sz in all_sizes]),
        xticklabelrotation = π/4,
        yticks  = [2.0^i for i in -2:7],
        ytickformat = ys -> [@sprintf("%g", y) for y in ys])

    for width in WIDTHS
        valid_sizes    = [sz for sz in all_sizes if sz >= width]
        total_elements = [sz^ndim for sz in valid_sizes]
        for dim in 1:ndim
            ts         = [median(timings[(ndim, width, dim, sz)]) for sz in valid_sizes]
            throughput = total_elements ./ ts ./ 1e9
            lines!(ax, Float64.(valid_sizes), throughput;
                color     = width_col[width],
                linestyle = dim_style[dim],
                linewidth = 1.5)
            scatter!(ax, Float64.(valid_sizes), throughput;
                color      = width_col[width],
                markersize = 5)
        end
    end
end

# Legend: colour = stencil width, linestyle = differentiation direction
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
outfile = joinpath(outdir, "matmul.svg")
save(outfile, fig)
println("\nSaved → $outfile")
