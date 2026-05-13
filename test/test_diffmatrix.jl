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

    #  different width
    DA = DiffMatrix(xs, 3, 1)
    DB = DiffMatrix(xs, 5, 1)

    @test all(FDGrids.full(DA + DB) .== FDGrids.full(DA) + FDGrids.full(DB))

    #  different optimisation
    DA = DiffMatrix(xs, 3, 1; optimise=true)
    DB = DiffMatrix(xs, 3, 1; optimise=false)

    @test typeof(D + D) == DiffMatrix{Float64, 3, true}
end

@testset "test full.                                " begin
    M = 32
    width = 3

    # get grid
    xs = gridpoints(M, -1, 1, 0.5)

    # make grid from -1 to 1 using α = 0.5
    D1 = DiffMatrix(xs, width, 1)

    # get full matrix
    D1_full = full(D1)

    for i = 1:M, j = 1:M
        @test D1_full[i, j] == D1[i, j]
    end
end
