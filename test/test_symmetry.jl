@testset "left boundary symmetry output, centre on node" begin
    xs = collect(range(0.0, 1.0; length = 21))
    centre = 0.0
    width = 3
    H = width >> 1

    ghosts = reverse(2centre .- xs[2:H+1])
    xext = vcat(ghosts, xs)
    Dext = DiffMatrix(xext, width, 1)

    u_even = cos.(π .* xs)
    uext_even = cos.(π .* xext)
    D_even = DiffMatrix(xs, width, 1; symmetry = (EvenSymmetry(centre), NoSymmetry()))
    du_even = D_even * u_even
    du_even_ext = Dext * uext_even
    @test du_even[1:H] ≈ du_even_ext[H+1:2H] atol = 1e-12

    u_odd = sin.(π .* xs)
    uext_odd = sin.(π .* xext)
    D_odd = DiffMatrix(xs, width, 1; symmetry = (OddSymmetry(centre), NoSymmetry()))
    du_odd = D_odd * u_odd
    du_odd_ext = Dext * uext_odd
    @test du_odd[1:H] ≈ du_odd_ext[H+1:2H] atol = 1e-12
end

@testset "left boundary symmetry output, centre off node" begin
    xs = collect(range(0.1, 1.0; length = 21))
    centre = 0.0
    width = 3
    H = width >> 1

    ghosts = reverse(2centre .- xs[1:H])
    xext = vcat(ghosts, xs)
    Dext = DiffMatrix(xext, width, 1)

    u_even = cos.(π .* xs)
    uext_even = cos.(π .* xext)
    D_even = DiffMatrix(xs, width, 1; symmetry = (EvenSymmetry(centre), NoSymmetry()))
    du_even = D_even * u_even
    du_even_ext = Dext * uext_even
    @test du_even[1:H] ≈ du_even_ext[H+1:2H] atol = 1e-12

    u_odd = sin.(π .* xs)
    uext_odd = sin.(π .* xext)
    D_odd = DiffMatrix(xs, width, 1; symmetry = (OddSymmetry(centre), NoSymmetry()))
    du_odd = D_odd * u_odd
    du_odd_ext = Dext * uext_odd
    @test du_odd[1:H] ≈ du_odd_ext[H+1:2H] atol = 1e-12
end

@testset "right boundary symmetry output, centre on node" begin
    xs = collect(range(0.0, 1.0; length = 21))
    centre = last(xs)
    width = 3
    H = width >> 1
    N = length(xs)

    ghosts = 2centre .- reverse(xs[N-H:N-1])
    xext = vcat(xs, ghosts)
    Dext = DiffMatrix(xext, width, 1)

    u_even = cos.(π .* xs)
    uext_even = cos.(π .* xext)
    D_even = DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), EvenSymmetry(centre)))
    du_even = D_even * u_even
    du_even_ext = Dext * uext_even
    @test du_even[N-H+1:N] ≈ du_even_ext[N-H+1:N] atol = 1e-12

    u_odd = sin.(π .* xs)
    uext_odd = sin.(π .* xext)
    D_odd = DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), OddSymmetry(centre)))
    du_odd = D_odd * u_odd
    du_odd_ext = Dext * uext_odd
    @test du_odd[N-H+1:N] ≈ du_odd_ext[N-H+1:N] atol = 1e-12
end

@testset "right boundary symmetry output, centre off node" begin
    xs = collect(range(0.0, 0.9; length = 21))
    centre = 1.0
    width = 3
    H = width >> 1
    N = length(xs)

    ghosts = 2centre .- reverse(xs[N-H+1:N])
    xext = vcat(xs, ghosts)
    Dext = DiffMatrix(xext, width, 1)

    u_even = cos.(π .* xs)
    uext_even = cos.(π .* xext)
    D_even = DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), EvenSymmetry(centre)))
    du_even = D_even * u_even
    du_even_ext = Dext * uext_even
    @test du_even[N-H+1:N] ≈ du_even_ext[N-H+1:N] atol = 1e-12

    u_odd = sin.(π .* xs)
    uext_odd = sin.(π .* xext)
    D_odd = DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), OddSymmetry(centre)))
    du_odd = D_odd * u_odd
    du_odd_ext = Dext * uext_odd
    @test du_odd[N-H+1:N] ≈ du_odd_ext[N-H+1:N] atol = 1e-12
end
