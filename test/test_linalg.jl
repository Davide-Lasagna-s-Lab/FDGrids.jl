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
            D = DiffMatrix(xs, width, 2; optimise=false)
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

            # solve with OPTIMISE=false and compare to the full LU reference
            b = randn(M)
            x_banded = ldiv!(luD2, copy(b))
            x_full   = ldiv!(luDfull, copy(b))
            @test norm(x_banded - x_full)/M < 1e-12
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

@testset "boundary value problem example            " begin
    M = 64
    g = grid(M, -1, 1, GaussLobattoGrid())

    L = DiffMatrix(g.xs, 7, 2)
    L[1,   :] .= basis_vector(1, M)
    L[end, :] .= basis_vector(M, M)

    rhs = -(π^2) .* sin.(π .* g.xs)
    rhs[1]   = 0
    rhs[end] = 0

    u = ldiv!(lu!(L), copy(rhs))

    @test maximum(abs, u .- sin.(π .* g.xs)) < 1e-6
    @test u[1] ≈ 0 atol=1e-14
    @test u[end] ≈ 0 atol=1e-14
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
