using InteractiveUtils
using BenchmarkTools
using LinearAlgebra
using FDGrids

width = 7
P = 121
D = DiffMatrix(gridpoints(P, -1, 1), width, 1)

x = rand(P)
y = rand(P)

@btime mul!($y, $D, $x)

x = rand(P, 20, 20)
y = rand(P, 20, 20)

@btime mul!($y, $D, $x)


# @test norm(Df * u .- D * u)/50 < 1e-14

    #     T_mymul = @belapsed mul!($y, $A_mymul, $x)

        
        # A_blas  = CoeffMatrix(to_banded_format(data))

    # BLAS.set_num_threads(1)
    # N1, N2, N3 = (128, 128, 32)

    # for width = 3:2:31
    #     # get data
    #     data = rand(width, N3)

    #     A_mymul = CoeffMatrix(data)
    #     A_blas  = CoeffMatrix(to_banded_format(data))

    #     x = rand(N1, N2, N3) + im*rand(N1, N2, N3)
    #     y = similar(x)
    #     buff1 = zeros(N3)
    #     buff2 = zeros(N3)
    #     buff3 = zeros(N3)

    #     # check results are correct
    #     y_mymul = mul!(similar(x), A_mymul, x)
    #     y_blas  = blas_mul!(similar(x), A_blas, x, buff1, buff2, buff3)
    #     # @assert norm(y_mymul .- y_blas) < 1e-10
    #     println(norm(y_mymul .- y_blas))

    #     T_mymul = @belapsed mul!($y, $A_mymul, $x)
    #     T_blas  = @belapsed blas_mul!($y, $A_blas, $x, $buff1, $buff2, $buff3)
    #     @printf "%02d : %.6f %.6f - %.6f\n" width 1e3*T_mymul 1e3*T_blas T_blas/T_mymul
    # end