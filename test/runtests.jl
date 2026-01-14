using LinearAlgebra
using FDGrids
using Test

@testset "test quadrature                           " begin

    # quadrature
    xs = range(-2, stop=2, length=121)
    I = exp(2) - exp(-2)
    @test abs(sum(exp.(xs) .* quadweights(xs, 1)) - I) < 1e-3
    @test abs(sum(exp.(xs) .* quadweights(xs, 2)) - I) < 5e-8
    @test abs(sum(exp.(xs) .* quadweights(xs, 3)) - I) < 2e-7
    @test abs(sum(exp.(xs) .* quadweights(xs, 4)) - I) < 1e-10
    @test abs(sum(exp.(reverse(collect(xs))) .* quadweights(reverse(collect(xs)), 4)) - I) < 1e-10

    # algo does not depend on points
    xs = [-1, -0.2, 0.2, 1]
    @test abs(sum((x->x^3).(xs) .* quadweights(xs, 3))) < 1e-5
    xs = [-1, -0.2, 0.1, 1]
    @test abs(sum((x->x^3).(xs) .* quadweights(xs, 3))) < 1e-5
    xs = [-1, -0.9, 0.1, 1]
    @test abs(sum((x->x^3).(xs) .* quadweights(xs, 3))) < 1e-5

    # trapz
    @test _quadweights([0, 1]) ≈ [1, 1]/2

    # simpson
    @test _quadweights([0, 0.5, 1]) ≈ [1, 4, 1]/6
    
    # generic polinomial
    xs = [-1, -0.2, 0.2, 1]
    @test sum((x->x^3).(xs) .* _quadweights(xs)) ≈ 0 atol=1e-15
    @test sum((x->x^2).(xs) .* _quadweights(xs)) ≈ 2/3
    @test sum((x->x  ).(xs) .* _quadweights(xs)) ≈ 0 atol=1e-15
    @test sum((x->1  ).(xs) .* _quadweights(xs)) ≈ 2

end

@testset "test grid                                 " begin
    # can't do silly things
    @test_throws ArgumentError gridpoints( 1, -1,  1, 1.0)
    @test_throws ArgumentError gridpoints(10, -1,  1, 2.0)
    @test_throws ArgumentError gridpoints(10, -1,  1, 0.0)
    @test_throws ArgumentError gridpoints(10,  1, -1, 0.5)

    # uniform distribution
    g = gridpoints(3, -2, 3, 1.0)
    
    # indexing 
    @test g[1] ≈ -2
    @test g[2] ≈  (3-2)/2
    @test g[3] ≈  3

    # for α tending to zero the points converge to the 
    # extrema of the chebychev polynomials to machine  accuracy
    for M in (10, 100, 1000)
        g = gridpoints(M, -1, 1, 1e-100)
        expected = reverse(cos.((0:(M-1))./(M-1).*π))
        @test maximum(abs.(g - expected)) < 1e-15
    end
end

@testset "test indexing                             " begin
    for M = 3:10
        for width = 3:2:9
            if M > width
                xs = gridpoints(M, -1, 1, 1)
        
                # diffmatrix
                D = DiffMatrix(xs, width, 1)

                # test
                # @test all(D .== full(D))
                fullD = FDGrids.full(D)
                for i = 1:M, j=1:M
                    @test D[i, j] == fullD[i, j]
                end
            end
        end
    end
end

@testset "slicing                                   " begin
    # points
    xs = gridpoints(10, -1, 1, 1)
        
    # diffmatrix
    D = DiffMatrix(xs, 3, 1)
    
    # set data
              D[1, :] .=  [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]
    @test all(D[1, :] .== [1, 2, 3, 0, 0, 0, 0, 0, 0, 0])

              D[end-1, :] .=  [0, 0, 0, 0, 0, 0, 4, 5, 6, 7]
    @test all(D[end-1, :] .== [0, 0, 0, 0, 0, 0, 0, 5, 6, 7])

              D[end, :] .=  [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    @test all(D[end, :] .== [0, 0, 0, 0, 0, 0, 0, 2, 3, 4])
end

@testset "algebra                                   " begin
    xs = gridpoints(6, -1, 1, 1)
        
    # diffmatrix
    D = DiffMatrix(xs, 3, 1)

    @test typeof(D)                             == DiffMatrix{Float64, 3, true}
    @test typeof(D + 3*I)                       == DiffMatrix{Float64, 3, true}
    @test typeof(D + D)                         == DiffMatrix{Float64, 3, true}
    @test typeof(D + 3*D)                       == DiffMatrix{Float64, 3, true}
    @test typeof(D + 3*Diagonal(rand(6))*D)     == DiffMatrix{Float64, 3, true}
    @test typeof(D + 2*D*(3*I))                 == DiffMatrix{Float64, 3, true}
    @test typeof(D + 3*im*Diagonal(rand(6))*D)  == DiffMatrix{Complex{Float64}, 3, true}

    #  different width
    DA = DiffMatrix(xs, 3, 1)
    DB = DiffMatrix(xs, 5, 1)
    C  = Diagonal(rand(6))

    @test all(FDGrids.full(DA + DB) .== FDGrids.full(DA) + FDGrids.full(DB))
    @test all(FDGrids.full(C*DA) .== C*FDGrids.full(DA))
    @test all(FDGrids.full(DA + 2*DA*(3*I)) .== FDGrids.full(DA) + 2*FDGrids.full(DA)*(3*I))
end

@testset "matvec/matmat product                     " begin
    for width = (3, 5, 7)
        D = DiffMatrix(gridpoints(50, -1, 1), width, 1)
        Df = FDGrids.full(D)
        
        u = rand(50)
        @test norm(Df * u .- D * u)/50 < 1e-14

        u = rand(50, 2)
        @test norm(Df * u .- D * u)/50 < 1e-14
    end
end

@testset "test diffmatrix at point                  " begin
    # number of points
    for M in (10, 20, 30)
        for width in (3, 5, 7)
            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D = DiffMatrix(xs, width, 1)

            # arrange to 3D array
            fs = exp.(1.0.*xs)

            for i = 1:M
                @test mul!(similar(fs), D, fs)[i] == mul!(D, fs, i)
            end
        end
    end
end

@testset "test diffmatrix 1st/2nd order - vec       " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7), 
                                                           (1.21, 3.22,  9.85), 
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs = exp.(1.0.*xs)

            # exact first derivative
            d1fs_EX = copy(fs)

            # exact second derivative
            d2fs_EX = copy(fs)
            
            # compute finite difference approximation along the first direction
            d1fs_FD      = similar(fs)
            d2fs_FD      = similar(fs)
            mul!(d1fs_FD, D1, fs)
            mul!(d2fs_FD, D2, fs)
           
            # the relative error should scale like M^{-o} where o = width-1
            v1 = maximum(abs.(d1fs_EX - d1fs_FD))/maximum(abs.(d1fs_EX))*M^(width-1)
            @test v1 < v1_max

            # for the second derivative we have the same order in the domain center
            i = M >> 1
            v2 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-1)
            @test v2 < v2_center_max

            # and one order less at the boundary
            i = 1
            v3 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-2)
            @test v3 < v2_bndr_max
        end
    end
end

@testset "test diffmatrix 1st/2nd order - mat       " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7), 
                                                           (1.21, 3.22,  9.85), 
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs          = zeros(M, 2)
            fs[:, 1] = exp.(1.0.*xs)
            fs[:, 2] = exp.(1.1.*xs)

            # exact first derivative
            d1fs_EX   = copy(fs)
            d1fs_EX[:, 1] .*= 1.0
            d1fs_EX[:, 2] .*= 1.1

            # exact second derivative
            d2fs_EX   = copy(fs)
            d2fs_EX[:, 1] .*= 1.0^2
            d2fs_EX[:, 2] .*= 1.1^2
            
            # compute finite difference approximation along the first direction
            d1fs_FD      = similar(fs)
            d2fs_FD      = similar(fs)
            mul!(d1fs_FD, D1, fs)
            mul!(d2fs_FD, D2, fs)

            # the relative error should scale like M^{-o} where o = width-1
            v1 = maximum(abs.(d1fs_EX - d1fs_FD))/maximum(abs.(d1fs_EX))*M^(width-1)
            @test v1 < v1_max

            # for the second derivative we have the same order in the domain center
            i = M >> 1
            v2 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-1)
            @test v2 < v2_center_max

            # and one order less at the boundary
            i = 1
            v3 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-2)
            @test v3 < v2_bndr_max
        end
    end
end

@testset "test diffmatrix 1st/2nd order - cube      " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7), 
                                                           (1.21, 3.22,  9.85), 
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs          = zeros(M, 2, 2)
            fs[:, 1, 1] = exp.(1.0.*xs)
            fs[:, 1, 2] = exp.(1.1.*xs)
            fs[:, 2, 1] = exp.(1.2.*xs)
            fs[:, 2, 2] = exp.(1.3.*xs)

            # exact first derivative
            d1fs_EX   = copy(fs)
            d1fs_EX[:, 1, 1] .*= 1.0
            d1fs_EX[:, 1, 2] .*= 1.1
            d1fs_EX[:, 2, 1] .*= 1.2
            d1fs_EX[:, 2, 2] .*= 1.3

            # exact second derivative
            d2fs_EX   = copy(fs)
            d2fs_EX[:, 1, 1] .*= 1.0^2
            d2fs_EX[:, 1, 2] .*= 1.1^2
            d2fs_EX[:, 2, 1] .*= 1.2^2
            d2fs_EX[:, 2, 2] .*= 1.3^2

            # compute finite difference approximation along the first direction
            d1fs_FD      = similar(fs)
            d2fs_FD      = similar(fs)
            mul!(d1fs_FD, D1, fs)
            mul!(d2fs_FD, D2, fs)

            # the relative error should scale like M^{-o} where o = width-1
            v1 = maximum(abs.(d1fs_EX - d1fs_FD))/maximum(abs.(d1fs_EX))#*M^(width-1)
            @test v1 < v1_max

            # for the second derivative we have the same order in the domain center
            i = M >> 1
            v2 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-1)
            @test v2 < v2_center_max

            # and one order less at the boundary
            i = 1
            v3 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-2)
            @test v3 < v2_bndr_max
        end
    end
end

@testset "test diffmatrix 1st/2nd order - hypercube " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7), 
                                                           (1.21, 3.22,  9.85), 
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs          = zeros(M, 2, 2, 2)
            fs[:, 1, 1, 1] = exp.(1.0.*xs)
            fs[:, 1, 2, 1] = exp.(1.1.*xs)
            fs[:, 2, 1, 1] = exp.(1.2.*xs)
            fs[:, 2, 2, 1] = exp.(1.3.*xs)
            fs[:, 1, 1, 2] = exp.(1.4.*xs)
            fs[:, 1, 2, 2] = exp.(1.5.*xs)
            fs[:, 2, 1, 2] = exp.(1.6.*xs)
            fs[:, 2, 2, 2] = exp.(1.7.*xs)

            # exact first derivative
            d1fs_EX   = copy(fs)
            d1fs_EX[:, 1, 1, 1] .*= 1.0
            d1fs_EX[:, 1, 2, 1] .*= 1.1
            d1fs_EX[:, 2, 1, 1] .*= 1.2
            d1fs_EX[:, 2, 2, 1] .*= 1.3
            d1fs_EX[:, 1, 1, 2] .*= 1.4
            d1fs_EX[:, 1, 2, 2] .*= 1.5
            d1fs_EX[:, 2, 1, 2] .*= 1.6
            d1fs_EX[:, 2, 2, 2] .*= 1.7

            # exact second derivative
            d2fs_EX   = copy(fs)
            d2fs_EX[:, 1, 1, 1] .*= 1.0^2
            d2fs_EX[:, 1, 2, 1] .*= 1.1^2
            d2fs_EX[:, 2, 1, 1] .*= 1.2^2
            d2fs_EX[:, 2, 2, 1] .*= 1.3^2
            d2fs_EX[:, 1, 1, 2] .*= 1.4^2
            d2fs_EX[:, 1, 2, 2] .*= 1.5^2
            d2fs_EX[:, 2, 1, 2] .*= 1.6^2
            d2fs_EX[:, 2, 2, 2] .*= 1.7^2

            # compute finite difference approximation along the first direction
            d1fs_FD      = similar(fs)
            d2fs_FD      = similar(fs)
            mul!(d1fs_FD, D1, fs)
            mul!(d2fs_FD, D2, fs)

            # the relative error should scale like M^{-o} where o = width-1
            v1 = maximum(abs.(d1fs_EX - d1fs_FD))/maximum(abs.(d1fs_EX))#*M^(width-1)
            @test v1 < v1_max

            # for the second derivative we have the same order in the domain center
            i = M >> 1
            v2 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-1)
            @test v2 < v2_center_max

            # and one order less at the boundary
            i = 1
            v3 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-2)
            @test v3 < v2_bndr_max
        end
    end
end

@testset "test diffmatrix with range - hypercube    " begin
    # the order of accuracy is the width minus one
    for (width, v1_max) in zip((3,    5,     7), 
                               (1.21, 3.22,  9.85))

        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # define range
            lwr       = rand(1 + (width >> 1):div(M, 2))
            upr       = rand(lwr + 1:M - (width >> 1))
            rng       = lwr:upr
            ghost_rng = (rng[1] - (width >> 1)):(rng[end] + (width >> 1))

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs = zeros(length(ghost_rng), 2, 2, 2)
            fs[:, 1, 1, 1] = exp.(1.0.*xs[ghost_rng])
            fs[:, 1, 2, 1] = exp.(1.1.*xs[ghost_rng])
            fs[:, 2, 1, 1] = exp.(1.2.*xs[ghost_rng])
            fs[:, 2, 2, 1] = exp.(1.3.*xs[ghost_rng])
            fs[:, 1, 1, 2] = exp.(1.4.*xs[ghost_rng])
            fs[:, 1, 2, 2] = exp.(1.5.*xs[ghost_rng])
            fs[:, 2, 1, 2] = exp.(1.6.*xs[ghost_rng])
            fs[:, 2, 2, 2] = exp.(1.7.*xs[ghost_rng])

            # exact first derivative
            d1fs_EX = copy(fs[(1 + (width >> 1)):(length(rng) + (width >> 1)), :, :, :])
            d1fs_EX[:, 1, 1, 1] .*= 1.0
            d1fs_EX[:, 1, 2, 1] .*= 1.1
            d1fs_EX[:, 2, 1, 1] .*= 1.2
            d1fs_EX[:, 2, 2, 1] .*= 1.3
            d1fs_EX[:, 1, 1, 2] .*= 1.4
            d1fs_EX[:, 1, 2, 2] .*= 1.5
            d1fs_EX[:, 2, 1, 2] .*= 1.6
            d1fs_EX[:, 2, 2, 2] .*= 1.7

            # compute finite difference approximation along the first dirextion within range
            d1fs_FD = similar(fs[(1 + (width >> 1)):(length(rng) + (width >> 1)), :, :, :])
            d2fs_FD = similar(fs[(1 + (width >> 1)):(length(rng) + (width >> 1)), :, :, :])
            mul!(d1fs_FD, D1, fs, rng)

            # the relative error should scale like M^{-o} where o = width-1
            v1 = maximum(abs.(d1fs_EX - d1fs_FD))/maximum(abs.(d1fs_EX))#*M^(width-1)
            @test v1 < v1_max
        end
    end
end

@testset "test lufact                               " begin
    xs = gridpoints(12, -1, 1, 1)
            
    for width in (3, 5, 7)
        # diffmatrix
        D = DiffMatrix(xs, width, 2)

        # set boundary conditions
        D[  1, :] .= 0; D[  1,   1] = 1
        D[end, :] .= 0; D[end, end] = 1

        # just factorize
        luDfull = lu(FDGrids.full(D))
        luDdiff = lu(D)

        x = randn(length(xs))
        x_diff = ldiv!(luDdiff, copy(x))
        x_full = ldiv!(luDfull, copy(x))

        @test norm(x_diff - x_full) < 1e-13
    end
end

@testset "test new lufact                           " begin
    for M in (10, 100)
        xs = gridpoints(M, -1, 1, 1)
        for width in (3, 5, 7, 9)
            # diffmatrix
            # test fixed to optimise=false to facilitate comparison
            D = DiffMatrix(xs, width, 2, false)
            D[1,   :] .= [1, zeros(M-1)...]
            D[end, :] .= [zeros(M-1)..., 1];

            # just factorize full matrix withouth pivoting and reconstruct
            luDfull = lu(FDGrids.full(D), NoPivot())
            luD1 = luDfull.L + luDfull.U - Diagonal(ones(length(xs)))

            # factorise diffmatrix without inverting the diagonal element
            luD2 = lu!(D)

            # since they are the same, the factorisation does not spoil 
            # the structure of the differentiation matrices and we do not 
            # require additional storage
            @test norm(luD1 - full(luD2)) < 1e-8
            # @printf "%04d %04d %.5e\n" M width norm(luD1 - full(luD2))/M
        end
    end
end

@testset "test demo linsolve                        " begin
    for M in (100, 300)
        xs = gridpoints(M, -1, 1, 1)
        for width in (3, 5, 7, 9, 11, 13)
            # diffmatrix
            D = DiffMatrix(xs, width, 2)
            D[1,   :] .= [1, zeros(M-1)...]
            D[end, :] .= [zeros(M-1)..., 1];

            # random right hand side
            b = randn(M)

            # factorize full matrix without pivoting
            luDfull = lu(full(D), NoPivot())
            x_full = ldiv!(luDfull, copy(b))

            # factorise and solve using optimised routines
            lu!(D)
            x_banded = ldiv!(D, copy(b))

            @test norm(x_full - x_banded)/M < 2e-13
            # @printf "%04d %04d %.5e\n" M width norm(x_full - x_banded)/M
        end
    end
end

@testset "test BVP                                  " begin
    # solve u'' + u' = 1, with u(-1) = 2, u(1) = 0
    
    # test different orders
    for (width, val_max) in ( (3, 1.45), (5, 3) )

        for M ∈ (10, 20, 40, 80, 160, 320)
            # points
            xs = gridpoints(M, -1, 1, 0.5)

            # differentiation matrices
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # matrix of the problem
            L = D2 + D1

            # set boundary conditions
            L[  1, :] .= 0; L[  1,   1] = 1
            L[end, :] .= 0; L[end, end] = 1

            # just factorize
            luL = lu!(L)

            x = ones(length(xs))
            x[1]   = 2
            x[end] = 0

            f_diff = ldiv!(luL, copy(x))

            # exact solution from wolframalpha
            e = exp(1)
            f(x) = (3 + e^2 - 4*exp(1 - x) + x - e^2 * x)/(1 - e^2)
            f_exact = f.(xs)

            # order of accuracy
            val = maximum(abs, f_exact - f_diff) * M^(width - 1)
            @test val < val_max
        end
    end
end