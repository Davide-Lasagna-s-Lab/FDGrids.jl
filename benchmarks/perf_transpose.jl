using FDGrids
using LinearAlgebra
using BenchmarkTools
using PyPlot
using Printf

# ================================================================
# Configuration
# ================================================================
const WIDTHs = [3, 5, 7]
const Ns     = [32, 64, 128, 256]   # size along the differentiated axis
const OTHER  = 8                    # fixed size along all other axes
const SAMPLES = 40
const EVALS   = 5

# ================================================================
# Benchmark loop — collect all results first
# ================================================================
# results[N_arr][WIDTH][DIM][N_idx] = (t_fwd, t_adj)
# We store nothing for cases where N ≤ 2*WIDTH (no valid adjoint)

println("Running benchmarks...")

results = Dict()   # (N_arr, WIDTH, DIM, N) => (t_fwd, t_adj)

for N_arr in 1:4, WIDTH in WIDTHs, DIM in 1:N_arr, (N_idx, N) in enumerate(Ns)
    N > 2 * WIDTH || continue

    xs    = range(0.0, 1.0; length=N)
    shape = ntuple(d -> d == DIM ? N : OTHER, N_arr)
    x     = randn(Float64, shape...)
    y     = similar(x)
    A     = DiffMatrix(collect(xs), WIDTH, 1)
    At    = adjoint(A)

    t_fwd = @belapsed mul!($y, $A,  $x, $(Val(DIM))) samples=SAMPLES evals=EVALS
    t_adj = @belapsed mul!($y, $At, $x, $(Val(DIM))) samples=SAMPLES evals=EVALS

    results[(N_arr, WIDTH, DIM, N)] = (t_fwd, t_adj)
    @printf "  N_arr=%d  WIDTH=%d  DIM=%d  N=%3d   fwd=%7.2f µs   adj=%7.2f µs   slowdown=%.2fx\n" N_arr WIDTH DIM N t_fwd*1e6 t_adj*1e6 t_adj/t_fwd
end

# ================================================================
# Figure layout:  rows = N_arr (1..4),  cols = WIDTH (3, 5, 7)
# Lines in each subplot = DIM values, x-axis = N
# ================================================================

const DIM_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # tab blue/orange/green/red
const DIM_MARKERS = ["o", "s", "^", "D"]
const DIM_LABELS  = ["DIM=1", "DIM=2", "DIM=3", "DIM=4"]

nrows = 4
ncols = length(WIDTHs)

fig, axes = subplots(nrows, ncols,
                     figsize = (4.0 * ncols, 3.2 * nrows),
                     sharex  = false,
                     sharey  = false)

fig.suptitle("Transpose vs forward matmul — slowdown (Aᵀ·x / A·x)\n" *
             "Array size: N along DIM, $OTHER along all other dims",
             fontsize = 13, y = 1.01)

for (row, N_arr) in enumerate(1:4)
    for (col, WIDTH) in enumerate(WIDTHs)
        ax = axes[row, col]

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9,
                   alpha=0.4, zorder=1)

        for DIM in 1:N_arr
            valid_Ns   = [N for N in Ns if N > 2 * WIDTH && haskey(results, (N_arr, WIDTH, DIM, N))]
            isempty(valid_Ns) && continue
            slowdowns  = [results[(N_arr, WIDTH, DIM, N)][2] /
                          results[(N_arr, WIDTH, DIM, N)][1]  for N in valid_Ns]

            ax.plot(valid_Ns, slowdowns,
                    color      = DIM_COLORS[DIM],
                    marker     = DIM_MARKERS[DIM],
                    label      = DIM_LABELS[DIM],
                    linewidth  = 1.8,
                    markersize = 6,
                    zorder     = 2)
        end

        ax.set_title("N_arr = $N_arr,  WIDTH = $WIDTH", fontsize = 9)
        ax.set_xscale("log", base = 2)
        ax.set_xticks(Ns)
        ax.set_xticklabels(string.(Ns), fontsize = 7)
        ax.tick_params(axis = "y", labelsize = 7)
        ax.grid(true, alpha = 0.25, linestyle = ":")
        ax.set_ylim(0.33, 3)

        # y-axis label on leftmost column only
        if col == 1
            ax.set_ylabel("slowdown (×)", fontsize = 8)
        end

        # x-axis label on bottom row only
        if row == nrows
            ax.set_xlabel("N (along DIM)", fontsize = 8)
        end

        # legend: top-right subplot of each row (avoid clutter elsewhere)
        if col == ncols
            ax.legend(fontsize = 7, loc = "upper right", framealpha = 0.8)
        end
    end
end

fig.tight_layout()
outpath = joinpath(@__DIR__, "transpose_slowdown.png")
savefig(outpath, dpi = 150, bbox_inches = "tight")
println("\nFigure saved to $outpath")
