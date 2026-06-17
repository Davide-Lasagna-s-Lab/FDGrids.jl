@testset "test indexing                             " begin
    for M = 3:10
        for width = 3:2:9
            if M > width
                xs, _ = grid(M, -1, 1, MappedGrid(1))

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
    xs, _ = grid(10, -1, 1, MappedGrid(1))

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
    xs, _ = grid(6, -1, 1, MappedGrid(1))

    # diffmatrix
    D = DiffMatrix(xs, 3, 1)

    @test typeof(D)       <: DiffMatrix{Float64, 3, true}
    @test typeof(D + 3*I) <: DiffMatrix{Float64, 3, true}
    @test typeof(D + D)   <: DiffMatrix{Float64, 3, true}
    @test typeof(D + 3*D) <: DiffMatrix{Float64, 3, true}

    #  different width
    DA = DiffMatrix(xs, 3, 1)
    DB = DiffMatrix(xs, 5, 1)

    @test all(FDGrids.full(DA + DB) .== FDGrids.full(DA) + FDGrids.full(DB))

    #  different optimisation
    DA = DiffMatrix(xs, 3, 1; optimise=true)
    DB = DiffMatrix(xs, 3, 1; optimise=false)

    @test typeof(D + D) <: DiffMatrix{Float64, 3, true}
end

_uniformscaling_broadcast_dotted!(A, D, θ₀, θ₁) = (A .= θ₀ .* D .- θ₁ .* I; A)
_uniformscaling_broadcast_scaled!(A, D, θ₀, θ₁) = (A .= θ₀ .* D .- θ₁ * I; A)
_uniformscaling_mask_right!(A, D, θ) = (A .= D .* (θ .* I); A)
_uniformscaling_mask_left!(A, D, θ) = (A .= (θ .* I) .* D; A)
_diagonal_broadcast!(A, D, C, θ) = (A .= θ .* D .+ C; A)
_diffmatrix_pair_broadcast!(A, DA, DB) = (A .= 2 .* DA .- 3 .* DB; A)

const REQUIRE_ZERO_BROADCAST_ALLOCATIONS = VERSION >= v"1.11"

@testset "compact structured broadcast              " begin
    for M in (10, 32), width in (3, 5, 7)
        xs, _ = grid(M, -1, 1, MappedGrid(1))
        D  = DiffMatrix(xs, width, 2)
        θ₀, θ₁ = 2.5, 1.3
        J = θ₁ * I
        C = Diagonal(range(1.0, 2.0; length=M))
        Dfull = FDGrids.full(D)
        Jfull = Matrix(J, M, M)
        Cfull = Matrix(C)

        @test FDGrids.full(D .+ J) ≈ Dfull + J
        @test FDGrids.full(J .+ D) ≈ J + Dfull
        @test FDGrids.full(D .- J) ≈ Dfull - J
        @test FDGrids.full(J .- D) ≈ J - Dfull
        @test FDGrids.full(D .* J) ≈ Dfull .* Jfull
        @test FDGrids.full(J .* D) ≈ Jfull .* Dfull
        @test D .* I isa DiffMatrix{Float64, width, true}
        @test D .* (im * I) isa DiffMatrix{ComplexF64, width, true}

        @test FDGrids.full(D .+ C) ≈ Dfull + C
        @test FDGrids.full(C .- D) ≈ C - Dfull
        @test FDGrids.full(D .* C) ≈ Dfull .* Cfull
        @test FDGrids.full(C .* D) ≈ Cfull .* Dfull

        DA = DiffMatrix(xs, 3, 1; optimise=true)
        DB = DiffMatrix(xs, width, 2; optimise=false)
        @test FDGrids.full(DA .+ DB) ≈ FDGrids.full(DA) .+ FDGrids.full(DB)
        @test FDGrids.full(DA .- DB) ≈ FDGrids.full(DA) .- FDGrids.full(DB)
        @test FDGrids.full(DA .* DB) ≈ FDGrids.full(DA) .* FDGrids.full(DB)
        @test DA .+ DB isa DiffMatrix{Float64, width, true}

        # reference: manual loop
        A_ref = similar(D)
        A_ref .= θ₀ .* D
        for i in 1:M; A_ref[i, i] -= θ₁; end

        # broadcast with UniformScaling via .* I
        A1 = similar(D)
        _uniformscaling_broadcast_dotted!(A1, D, θ₀, θ₁)
        @test FDGrids.full(A1) ≈ FDGrids.full(A_ref)
        if REQUIRE_ZERO_BROADCAST_ALLOCATIONS
            @test (@allocated _uniformscaling_broadcast_dotted!(A1, D, θ₀, θ₁)) == 0
        end

        # broadcast with UniformScaling via * I (scalar-times-I form)
        A2 = similar(D)
        _uniformscaling_broadcast_scaled!(A2, D, θ₀, θ₁)
        @test FDGrids.full(A2) ≈ FDGrids.full(A_ref)
        if REQUIRE_ZERO_BROADCAST_ALLOCATIONS
            @test (@allocated _uniformscaling_broadcast_scaled!(A2, D, θ₀, θ₁)) == 0
        end

        # elementwise multiplication by I keeps only the diagonal
        A3 = similar(D)
        _uniformscaling_mask_right!(A3, D, θ₁)
        @test FDGrids.full(A3) ≈ Dfull .* Jfull
        if REQUIRE_ZERO_BROADCAST_ALLOCATIONS
            @test (@allocated _uniformscaling_mask_right!(A3, D, θ₁)) == 0
        end

        A4 = similar(D)
        _uniformscaling_mask_left!(A4, D, θ₁)
        @test FDGrids.full(A4) ≈ Jfull .* Dfull
        if REQUIRE_ZERO_BROADCAST_ALLOCATIONS
            @test (@allocated _uniformscaling_mask_left!(A4, D, θ₁)) == 0
        end

        A5 = similar(D)
        _diagonal_broadcast!(A5, D, C, θ₀)
        @test FDGrids.full(A5) ≈ θ₀ .* Dfull .+ C
        if REQUIRE_ZERO_BROADCAST_ALLOCATIONS
            @test (@allocated _diagonal_broadcast!(A5, D, C, θ₀)) == 0
        end

        A6 = similar(DA .+ DB)
        _diffmatrix_pair_broadcast!(A6, DA, DB)
        @test FDGrids.full(A6) ≈ 2 .* FDGrids.full(DA) .- 3 .* FDGrids.full(DB)
        if REQUIRE_ZERO_BROADCAST_ALLOCATIONS
            @test (@allocated _diffmatrix_pair_broadcast!(A6, DA, DB)) == 0
        end

        # result type is preserved
        @test A1 isa DiffMatrix
        @test A2 isa DiffMatrix
        @test A3 isa DiffMatrix
        @test A4 isa DiffMatrix
        @test A5 isa DiffMatrix
        @test A6 isa DiffMatrix
    end

    D = DiffMatrix(grid(8, -1, 1, MappedGrid(1))[1], 3, 1)
    @test_throws ArgumentError D .+ rand(8, 8)
    @test_throws ArgumentError rand(8, 8) .+ D
    @test_throws ArgumentError D .+ 1
    @test_throws ArgumentError 1 .- D
    @test_throws ArgumentError D .+ rand(8)
    @test_throws DimensionMismatch D .+ DiffMatrix(grid(9, -1, 1, MappedGrid(1))[1], 3, 1)
    @test_throws DimensionMismatch D .+ Diagonal(rand(9))
end

@testset "test full.                                " begin
    M = 32
    width = 3

    # get grid
    xs, _ = grid(M, -1, 1, MappedGrid(0.5))

    # make grid from -1 to 1 using α = 0.5
    D1 = DiffMatrix(xs, width, 1)

    # get full matrix
    D1_full = full(D1)

    for i = 1:M, j = 1:M
        @test D1_full[i, j] == D1[i, j]
    end
end
