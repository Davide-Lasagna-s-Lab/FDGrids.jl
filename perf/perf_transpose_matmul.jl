using FDGrids
using LinearAlgebra
using BenchmarkTools
using Printf


WIDTHs = [3, 5, 7]
Ns = [32, 64, 128, 256]
dim = 1

for WIDTH in WIDTHs, N in Ns
    sizes = (N, N)
    x = randn(Float64, sizes...)
    y = similar(x)
    xs = range(0.0, 1.0; length=sizes[dim])
    A  = DiffMatrix(collect(xs), WIDTH, 1)
    At = LinearAlgebra.Adjoint(A)

    bt = @btime mul!($y, $At, $x, $(Val(dim)))  samples = 80 evals = 10
    b  = @btime mul!($y, $A,  $x, $(Val(dim)))  samples = 80 evals = 10
end