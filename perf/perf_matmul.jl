using FDGrids
using LinearAlgebra
using BenchmarkTools
using Printf

# ================================================================
#  Configuration
# ================================================================

const SIZES = [32, 64, 128]
const WIDTH = 5
const ORDER = 1

# ================================================================
#  Helpers
# ================================================================

fmt(t) = t < 1e3 ? @sprintf("%8.1f ns", t) :
         t < 1e6 ? @sprintf("%8.2f μs", t/1e3) :
                    @sprintf("%8.3f ms", t/1e6)

function make_inputs(sizes, dim, ::Type{T}=Float64) where T
    x = randn(T, sizes...)
    y = similar(x)
    xs = range(0.0, 1.0; length=sizes[dim])
    A  = DiffMatrix(collect(xs), WIDTH, ORDER)
    return y, A, x
end

function bench(sizes, dim)
    y, A, x = make_inputs(sizes, dim)
    b = @benchmark mul!($y, $A, $x, $(Val(dim))) samples=400 evals=10
    return median(b).time
end

# ================================================================
#  Run
# ================================================================

function run()
    println()
    println("  branch : ", strip(read(`git rev-parse --abbrev-ref HEAD`, String)))
    println("  commit : ", strip(read(`git rev-parse --short HEAD`, String)))
    println("  julia  : ", VERSION)
    println()

    @printf("  %-22s  %-5s  %s\n", "array", "dim", "median time")
    println("  ", "-"^42)

    for N in 1:4
        for sz in SIZES
            sizes = ntuple(_ -> sz, N)
            for dim in 1:N
                t     = bench(sizes, dim)
                label = join(string.(sizes), "×")
                @printf("  %-22s  %d      %s\n", label, dim, fmt(t))
            end
            println()
        end
    end
end

run()
