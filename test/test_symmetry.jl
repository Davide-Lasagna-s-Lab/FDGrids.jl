@testset "left symmetry from extended grid, centre on node" begin
    xs = collect(range(0.0, 1.0; length = 21))
    centre = first(xs)
    width = 5
    H = width >> 1
    N = length(xs)

    ghosts = reverse(2centre .- xs[2:H+1])
    xext = vcat(ghosts, xs)
    Dext = Matrix(DiffMatrix(xext, width, 1))

    for symmetry in (EvenSymmetry(centre), OddSymmetry(centre))
        sign = symmetry isa EvenSymmetry ? 1 : -1
        Dsym = Matrix(DiffMatrix(xs, width, 1; symmetry = (symmetry, NoSymmetry())))

        for i in 1:H
            row = zeros(N)
            row_ext = Dext[i + H, :]
            row[2:H+1] .+= sign .* reverse(row_ext[1:H])   # folded ghost columns
            row        .+= row_ext[H+1:H+N]                # real columns
            @test Dsym[i, :] ≈ row atol = 1e-12
        end
    end
end

@testset "left symmetry from extended grid, centre off node" begin
    xs = collect(range(0.1, 1.0; length = 21))
    centre = 0.0
    width = 5
    H = width >> 1
    N = length(xs)

    ghosts = reverse(2centre .- xs[1:H])
    xext = vcat(ghosts, xs)
    Dext = Matrix(DiffMatrix(xext, width, 1))

    for symmetry in (EvenSymmetry(centre), OddSymmetry(centre))
        sign = symmetry isa EvenSymmetry ? 1 : -1
        Dsym = Matrix(DiffMatrix(xs, width, 1; symmetry = (symmetry, NoSymmetry())))

        for i in 1:H
            row = zeros(N)
            row_ext = Dext[i + H, :]
            row[1:H] .+= sign .* reverse(row_ext[1:H])     # folded ghost columns
            row      .+= row_ext[H+1:H+N]                  # real columns
            @test Dsym[i, :] ≈ row atol = 1e-12
        end
    end
end

@testset "right symmetry from extended grid, centre on node" begin
    xs = collect(range(0.0, 1.0; length = 21))
    centre = last(xs)
    width = 5
    H = width >> 1
    N = length(xs)

    ghosts = reverse(2centre .- xs[N-H:N-1])
    xext = vcat(xs, ghosts)
    Dext = Matrix(DiffMatrix(xext, width, 1))

    for symmetry in (EvenSymmetry(centre), OddSymmetry(centre))
        sign = symmetry isa EvenSymmetry ? 1 : -1
        Dsym = Matrix(DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), symmetry)))

        for i in 1:H
            row = zeros(N)
            row_ext = Dext[N-H+i, :]
            row[N-H:N-1] .+= sign .* reverse(row_ext[N+1:N+H])   # folded ghost columns
            row          .+= row_ext[1:N]                        # real columns
            @test Dsym[N-H+i, :] ≈ row atol = 1e-12
        end
    end
end

@testset "right symmetry from extended grid, centre off node" begin
    xs = collect(range(0.0, 0.9; length = 21))
    centre = 1.0
    width = 5
    H = width >> 1
    N = length(xs)

    ghosts = reverse(2centre .- xs[N-H+1:N])
    xext = vcat(xs, ghosts)
    Dext = Matrix(DiffMatrix(xext, width, 1))

    for symmetry in (EvenSymmetry(centre), OddSymmetry(centre))
        sign = symmetry isa EvenSymmetry ? 1 : -1
        Dsym = Matrix(DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), symmetry)))

        for i in 1:H
            row = zeros(N)
            row_ext = Dext[N-H+i, :]
            row[N-H+1:N] .+= sign .* reverse(row_ext[N+1:N+H])   # folded ghost columns
            row          .+= row_ext[1:N]                        # real columns
            @test Dsym[N-H+i, :] ≈ row atol = 1e-12
        end
    end
end
