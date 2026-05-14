# Benchmark: DiffMatrix generated lu! vs generic AbstractMatrix lu!
#
# Compares two factorisation paths for the same banded system:
#   Generic   — _banded_lu!(copy(D), p, q) on the DiffMatrix via getindex/setindex!
#   DiffMatrix — lu!(copy(D)) via the @generated compact kernel with direct coeffs access
#
# Uses a second-order differential operator so that the matrix has non-zero
# diagonal entries in all interior rows and no-pivot LU is well-defined.
#
# Run:
#   julia --project=benchmarks benchmarks/lu_factor.jl

using BenchmarkTools, LinearAlgebra, FDGrids, Printf

const WIDTHS  = [3, 5, 7]
const SIZES   = [16, 32, 64, 128, 256, 512, 1024, 2048]
const SAMPLES = 200

function time_factor(N, width)
    xs = collect(range(-1.0, 1.0; length=N))
    D  = DiffMatrix(xs, width, 2)
    D[1,   :] .= basis_vector(1, N)
    D[end, :] .= basis_vector(N, N)

    WD = width - 1

    # Verify both paths give the same solution on a random rhs
    rhs = randn(N)
    D_gen = copy(D); FDGrids._banded_lu!(D_gen, WD, WD; check=false)
    x_gen     = ldiv!(BandedMatrixLU(D_gen, WD, WD), copy(rhs))
    x_compact = ldiv!(lu!(copy(D)), copy(rhs))
    err = norm(x_gen - x_compact)
    err < 1e-10 * N ||
        error("solve mismatch at N=$N width=$width  err=$err")

    # Generic: AbstractMatrix path through getindex/setindex! on DiffMatrix
    t_gen = median(@benchmark FDGrids._banded_lu!(copy($D), $WD, $WD; check=false) samples=SAMPLES evals=3).time * 1e-9

    # Compact DiffMatrix: @generated kernel with direct coeffs access
    t_compact = median(@benchmark lu!(copy($D)) samples=SAMPLES evals=3).time * 1e-9

    return t_gen, t_compact
end

println("Benchmarking LU factorisation: Generic (getindex) vs @generated (direct coeffs)")
println("width  N      Generic (μs)  DiffMatrix (μs)  Speedup")
println("─"^60)

for w in WIDTHS
    for N in SIZES
        N < w && continue
        tg, tc = time_factor(N, w)
        @printf "  %d  %5d   %10.2f    %10.2f      %.2fx\n" w N tg*1e6 tc*1e6 tg/tc
    end
    println()
end
