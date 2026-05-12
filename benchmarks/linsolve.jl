# Benchmark: FDGrids compact linear solve vs LAPACK banded LU.
#
# Three solve paths are compared:
#   LAPACK   — lu(D) → DiffMatrixLU → ldiv! via dgbtrf!/dgbtrs!
#   FDGrids  — lu!(copy(D)) → DiffMatrix (factorised in-place) → @generated ldiv!
#   Generic  — same factorisation but dispatched through plain scalar loops
#
# Sweeps over grid sizes and stencil widths 3, 5, 7 (width = number of stencil points).
# Saves linsolve_speedup.svg to ../docs/src/assets/benchmarks/.
#
# Setup (once):
#   julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#   julia --project=benchmarks benchmarks/linsolve.jl

using BenchmarkTools, LinearAlgebra, FDGrids, CairoMakie, Printf

const WIDTHS  = [3, 5, 7]
const SIZES   = [16, 32, 64, 128, 256, 512, 1024, 2048]
const SAMPLES = 300

function time_solve(N, width)
    xs = collect(range(-1.0, 1.0; length=N))
    D  = DiffMatrix(xs, width, 1)
    # Replace boundary rows with identity to enforce Dirichlet BCs and make D non-singular.
    D[1,   :] .= [1; zeros(N - 1)]
    D[end, :] .= [zeros(N - 1); 1]
    b  = randn(N)

    # LAPACK pivoted banded LU
    lu_lapack = lu(D)
    t_lapack  = @belapsed ldiv!($lu_lapack, bc) setup=(bc=copy($b)) samples=SAMPLES evals=5

    # FDGrids compact @generated solve
    lu_fd = lu!(copy(D))
    t_fd  = @belapsed ldiv!($lu_fd, bc) setup=(bc=copy($b)) samples=SAMPLES evals=5

    # Generic scalar reference (same factorised storage, different dispatch)
    lu_gen  = lu!(copy(D))
    WD      = width - 1
    t_gen   = @belapsed ldiv!($lu_gen, bc, $WD, $WD) setup=(bc=copy($b)) samples=SAMPLES evals=5

    return t_lapack, t_fd, t_gen
end

println("Benchmarking linear solves…")
results = Dict{Int, NamedTuple}()
for w in WIDTHS
    t_lapack = Float64[]
    t_fd     = Float64[]
    t_gen    = Float64[]
    sizes_w  = Int[]
    for N in SIZES
        N < w && continue
        print("  width=$w  N=$N … ")
        tl, tf, tg = time_solve(N, w)
        push!(t_lapack, tl); push!(t_fd, tf); push!(t_gen, tg); push!(sizes_w, N)
        @printf "LAPACK %.2f μs  FDGrids %.2f μs  Generic %.2f μs\n" tl*1e6 tf*1e6 tg*1e6
    end
    results[w] = (sizes=sizes_w, lapack=t_lapack, fdgrids=t_fd, generic=t_gen)
end

# ─── Figure ───────────────────────────────────────────────────────────────────
colors    = Makie.wong_colors()
width_col = Dict(3 => colors[1], 5 => colors[2], 7 => colors[3])

fig = Figure(size=(900, 380))

# Left panel: absolute time (μs) for all three methods
ax1 = Axis(fig[1, 1];
    xlabel     = "Grid size N",
    ylabel     = "Solve time (μs)",
    title      = "Absolute solve time",
    xscale     = log2,
    yscale     = log10,
    xticks     = SIZES,
    xticklabelrotation = π/4)

# Right panel: speedup of FDGrids over LAPACK
ax2 = Axis(fig[1, 2];
    xlabel     = "Grid size N",
    ylabel     = "Speedup over LAPACK",
    title      = "FDGrids speedup vs LAPACK",
    xscale     = log2,
    xticks     = SIZES,
    xticklabelrotation = π/4)

hlines!(ax2, [1.0]; linestyle=:dash, color=:black, linewidth=1)

for w in WIDTHS
    r   = results[w]
    col = width_col[w]
    lab = "width $w"

    lines!(ax1, r.sizes, r.lapack  .* 1e6; color=col, linestyle=:solid,  linewidth=2, label="$lab LAPACK")
    lines!(ax1, r.sizes, r.fdgrids .* 1e6; color=col, linestyle=:dash,   linewidth=2, label="$lab FDGrids")
    lines!(ax1, r.sizes, r.generic .* 1e6; color=col, linestyle=:dot,    linewidth=2, label="$lab Generic")

    speedup = r.lapack ./ r.fdgrids
    lines!(ax2,   r.sizes, speedup; color=col, linewidth=2, label=lab)
    scatter!(ax2, r.sizes, speedup; color=col, markersize=6)
end

# Shared legend entries for line styles (left panel)
elem_solid  = LineElement(color=:black, linestyle=:solid,  linewidth=2)
elem_dashed = LineElement(color=:black, linestyle=:dash,   linewidth=2)
elem_dotted = LineElement(color=:black, linestyle=:dot,    linewidth=2)
style_labels = ["LAPACK", "FDGrids", "Generic"]

# Width legend (both panels use color)
width_elems = [LineElement(color=width_col[w], linewidth=2) for w in WIDTHS]

Legend(fig[2, 1],
    [elem_solid, elem_dashed, elem_dotted],
    style_labels;
    orientation=:horizontal, framevisible=false, tellwidth=false)

Legend(fig[2, 2],
    width_elems,
    ["width $w" for w in WIDTHS];
    orientation=:horizontal, framevisible=false, tellwidth=false)

outdir = joinpath(@__DIR__, "..", "docs", "src", "assets", "benchmarks")
mkpath(outdir)
outfile = joinpath(outdir, "linsolve_speedup.svg")
save(outfile, fig)
println("\nSaved → $outfile")
