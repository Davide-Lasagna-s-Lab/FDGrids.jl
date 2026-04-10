using FDGrids
using LinearAlgebra
using BenchmarkTools
using Printf

# # ================================================================
# #  Configuration
# # ================================================================

# const SIZES = [32, 64, 128]
# const WIDTH = 5
# const ORDER = 1

# # ================================================================
# #  Helpers
# # ================================================================

# fmt(t) = t < 1e3 ? @sprintf("%8.1f ns", t) :
#          t < 1e6 ? @sprintf("%8.2f μs", t/1e3) :
#                     @sprintf("%8.3f ms", t/1e6)

# function make_inputs(sizes, dim, ::Type{T}=Float64) where T

WIDTH = 5
ORDER = 1

sizes = (32, 32, 32, 32)
dim = 4
x = randn(Float64, sizes...)
y = similar(x)
xs = range(0.0, 1.0; length=sizes[dim])
A  = DiffMatrix(collect(xs), WIDTH, ORDER)

At = LinearAlgebra.Adjoint(A)
println(@elapsed mul!(y, At, x, Val(dim)), )

println(@elapsed mul!(y, A, x, Val(dim)))

# bt = @btime mul!($y, $At, $x, $(Val(dim)))  samples = 40 evals = 1
b  = @btime mul!($y, $A,  $x, $(Val(dim)))  samples = 40 evals = 1




    # return y, A, x
# end

# function bench(sizes, dim)
#     y, A, x = make_inputs(sizes, dim)
#     b = @benchmark mul!($y, $A, $x, $(Val(dim))) samples=400 evals=10
#     return median(b).time
# end

# ================================================================
#  Run
# ================================================================

# function run()
#     println()
#     println("  branch : ", strip(read(`git rev-parse --abbrev-ref HEAD`, String)))
#     println("  commit : ", strip(read(`git rev-parse --short HEAD`, String)))
#     println("  julia  : ", VERSION)
#     println()

#     @printf("  %-22s  %-5s  %s\n", "array", "dim", "median time")
#     println("  ", "-"^42)

#     for N in 1:4
#         for sz in SIZES
#             sizes = ntuple(_ -> sz, N)
#             for dim in 1:N
#                 t     = bench(sizes, dim)
#                 label = join(string.(sizes), "×")
#                 @printf("  %-22s  %d      %s\n", label, dim, fmt(t))
#             end
#             println()
#         end
#     end
# end

# run()
