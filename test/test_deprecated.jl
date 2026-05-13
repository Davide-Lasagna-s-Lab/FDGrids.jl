@testset "test quadrature (old interface)           " begin

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

    xs_new = gridpoints(64, -1.0, 2.0, 0.5)
    xs_old, _ = grid(64, -1.0, 2.0, MappedGrid(0.5))
    @test xs_new ≈ xs_old

    xs_new2 = gridpoints(64, -1.0, 2.0, GaussLobattoGrid())
    xs_old2, _ = grid(64, -1.0, 2.0, GaussLobattoGrid())
    @test xs_new2 ≈ xs_old2
end
