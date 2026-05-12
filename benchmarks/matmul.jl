# Benchmark: DiffMatrix mul! across array dimensions 1–4, stencil widths 3/5/7,
# and all differentiation directions.
#
# For each array dimension N (1,2,3,4) and stencil width, the benchmark measures
# mul!(y, D, x, Val(dim)) for dim = 1…N and a range of per-axis grid sizes.
# Arrays are square (same size along every axis).
#
# Saves matmul.svg to ../docs/src/assets/benchmarks/.
#
# Setup (once):
#   julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#   julia --project=benchmarks benchmarks/matmul.jl

using BenchmarkTools, LinearAlgebra, FDGrids, CairoMakie

# Per-axis grid sizes for each array dimension.
# Total elements grow as size^N, so we shrink the range for higher N.
const SIZES_BY_NDIM = Dict(
    1 => [32, 64, 128, 256, 512, 1024],
    2 => [16, 32, 64, 128, 256],
    3 => [8,  16, 32, 64,  128],
    4 => [8,  12, 16, 24,  32],
)
const WIDTHS   = [3, 5, 7]
const ORDER    = 1
const SAMPLES  = 200

function time_mul(sz, ndim, dim, width)
    sizes = ntuple(_ -> sz, ndim)
    xs    = collect(range(0.0, 1.0; length=sz))
    D     = DiffMatrix(xs, width, ORDER)
    x     = randn(sizes...)
    y     = similar(x)
    return @belapsed mul!($y, $D, $x, $(Val(dim))) samples=SAMPLES evals=3
end

# ─── Collect timings ─────────────────────────────────────────────────────────
println("Benchmarking mul!…")

# results[ndim][width][dim] = Vector{Float64}  (one entry per size)
results = Dict{Int, Dict{Int, Dict{Int, Vector{Float64}}}}()

for ndim in 1:4
    results[ndim] = Dict()
    for width in WIDTHS
        results[ndim][width] = Dict()
        for dim in 1:ndim
            results[ndim][width][dim] = Float64[]
        end
    end
    for sz in SIZES_BY_NDIM[ndim]
        total = sz^ndim
        for width in WIDTHS
            for dim in 1:ndim
                print("  $(ndim)D  sz=$sz  width=$width  dim=$dim … ")
                t = time_mul(sz, ndim, dim, width)
                push!(results[ndim][width][dim], t)
                @printf "%.2f μs  (%.2f GEl/s)\n" t*1e6 total/t/1e9
            end
        end
    end
end

# ─── Figure ──────────────────────────────────────────────────────────────────
colors    = Makie.wong_colors()
width_col = Dict(3 => colors[1], 5 => colors[2], 7 => colors[3])
dim_style = [:solid, :dash, :dashdot, :dot]

fig = Figure(size=(1100, 900))
Label(fig[0, :], "mul! throughput — DiffMatrix applied along each array axis";
      fontsize=14, font=:bold)

for ndim in 1:4
    row = cld(ndim, 2)
    col = isodd(ndim) ? 1 : 2
    sizes = SIZES_BY_NDIM[ndim]
    total_elements = [sz^ndim for sz in sizes]

    ax = Axis(fig[row, col];
        title   = "$(ndim)D array",
        xlabel  = "Total elements",
        ylabel  = "Throughput (GEl/s)",
        xscale  = log10,
        yscale  = log10,
        xtickformat = xs -> [@sprintf("%.0e", x) for x in xs])

    for width in WIDTHS
        for dim in 1:ndim
            ts = results[ndim][width][dim]
            throughput = total_elements ./ ts ./ 1e9
            lines!(ax, total_elements, throughput;
                color     = width_col[width],
                linestyle = dim_style[dim],
                linewidth = 1.5)
            scatter!(ax, total_elements, throughput;
                color     = width_col[width],
                markersize = 5)
        end
    end
end

# Legend: colors = stencil width, linestyles = differentiation direction
width_elems = [LineElement(color=width_col[w], linewidth=2) for w in WIDTHS]
dim_elems   = [LineElement(color=:black, linestyle=dim_style[d], linewidth=2) for d in 1:4]

Legend(fig[3, 1],
    width_elems, ["width $w" for w in WIDTHS];
    orientation=:horizontal, framevisible=false, tellwidth=false,
    title="Stencil width")

Legend(fig[3, 2],
    dim_elems, ["dim $d" for d in 1:4];
    orientation=:horizontal, framevisible=false, tellwidth=false,
    title="Differentiation direction")

outdir = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(outdir)
outfile = joinpath(outdir, "matmul.svg")
save(outfile, fig)
println("\nSaved → $outfile")
