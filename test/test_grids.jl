@testset "test grid API                                  " begin
    M = 64
    l = -1.0
    h =  2.0
    I_exp = exp(h) - exp(l)   # ∫_l^h exp(x) dx, used throughout

    # ---- return type and named tuple fields ----
    for dist in (MappedGrid(0.5), MappedGrid(0.5, 2), UniformGrid(), GaussLobattoGrid())
        result = grid(M, l, h, dist)
        @test result isa NamedTuple
        @test haskey(result, :xs)
        @test haskey(result, :ws)
        @test length(result.xs) == M
        @test length(result.ws) == M
    end

    # ---- points are sorted ascending and within [l, h] ----
    for dist in (MappedGrid(0.5), UniformGrid(), GaussLobattoGrid())
        xs, _ = grid(M, l, h, dist)
        @test issorted(xs)
        @test xs[1]   ≈ l
        @test xs[end] ≈ h
    end

    # ---- weights sum to interval length (∫_l^h 1 dx = h - l) ----
    for dist in (MappedGrid(0.5), MappedGrid(0.5, 2), UniformGrid(), GaussLobattoGrid())
        _, ws = grid(M, l, h, dist)
        @test sum(ws) ≈ h - l  atol=1e-12
    end

    # ---- UniformGrid: weights always strictly positive ----
    _, ws = grid(M, l, h, UniformGrid())
    @test all(ws .> 0)

    # ---- GaussLobattoGrid: weights always strictly positive ----
    _, ws = grid(M, l, h, GaussLobattoGrid())
    @test all(ws .> 0)

    # ---- UniformGrid: correct trapezoidal structure ----
    # interior weights equal h_step, endpoints equal h_step/2
    M2    = 11
    xs, ws = grid(M2, 0.0, 1.0, UniformGrid())
    h_step = 1.0 / (M2 - 1)
    @test ws[1]   ≈ h_step / 2
    @test ws[end] ≈ h_step / 2
    @test all(ws[2:end-1] .≈ h_step)

    # ---- GaussLobattoGrid: Chebyshev-Lobatto nodes on [-1, 1] ----
    # nodes should match cos formula exactly
    xs, _ = grid(M, -1.0, 1.0, GaussLobattoGrid())
    expected = [cos(π * (M - 1 - j) / (M - 1)) for j in 0:M-1]
    @test xs ≈ expected  atol=1e-14

    # ---- MappedGrid order parameter ----
    # order=1 (trapezoidal) should be less accurate than order=4 for exp
    xs1, ws1 = grid(M, l, h, MappedGrid(0.5, 1))
    xs4, ws4 = grid(M, l, h, MappedGrid(0.5, 4))
    err1 = abs(sum(exp.(xs1) .* ws1) - I_exp)
    err4 = abs(sum(exp.(xs4) .* ws4) - I_exp)
    @test err4 < err1

    # ---- integration accuracy ----
    # UniformGrid: trapezoidal is O(h²), so error ∝ 1/M²
    # doubling M should reduce error by ~4
    _, ws_32  = grid(32,  l, h, UniformGrid())
    xs_32, _  = grid(32,  l, h, UniformGrid())
    _, ws_64  = grid(64,  l, h, UniformGrid())
    xs_64, _  = grid(64,  l, h, UniformGrid())
    err_32 = abs(sum(exp.(xs_32) .* ws_32) - I_exp)
    err_64 = abs(sum(exp.(xs_64) .* ws_64) - I_exp)
    @test err_32 / err_64 > 3.5   # close to 4× improvement

    # GaussLobattoGrid: spectral accuracy — 64 points should be essentially exact
    xs_gl, ws_gl = grid(64, l, h, GaussLobattoGrid())
    @test abs(sum(exp.(xs_gl) .* ws_gl) - I_exp) < 1e-14

    # GaussLobattoGrid: exact for polynomials of degree ≤ 2M-3
    # test with a degree-5 polynomial where M=8 is exact (2*8-3 = 13 ≥ 5)
    xs_gl, ws_gl = grid(8, -1.0, 1.0, GaussLobattoGrid())
    f5(x) = x^5 - 3x^3 + x    # exact integral on [-1,1] = 0
    @test abs(sum(f5.(xs_gl) .* ws_gl)) < 1e-14

    # ---- HalfChebyshevGrid ----
    # The weights already include the radial measure r dr, so using f(r)=r cos(r)
    # gives an even integrand r^2 cos(r) for the underlying symmetric rule.
    R = 2.0
    I_radial_rcosr = R^2 * sin(R) + 2R * cos(R) - 2sin(R) # ∫_0^R r^2 cos(r) dr
    g = grid(M, R, HalfChebyshevGrid())
    @test g isa NamedTuple
    @test haskey(g, :xs)
    @test haskey(g, :ws)
    @test length(g.xs) == M
    @test length(g.ws) == M
    @test issorted(g.xs)
    @test 0.0 < g.xs[1]
    @test g.xs[end] ≈ R
    @test all(g.ws .> 0)
    @test sum(g.xs .* cos.(g.xs) .* g.ws) ≈ I_radial_rcosr atol=1e-12

    r2 = grid(2M, -R, R, GaussLobattoGrid())
    @test g.xs ≈ r2.xs[M+1:end]
    @test g.ws ≈ r2.xs[M+1:end] .* r2.ws[M+1:end]

    # ---- error handling ----
    @test_throws ArgumentError grid(1, l, h, GaussLobattoGrid())   # M < 2
    @test_throws ArgumentError grid(M, h, l, GaussLobattoGrid())   # l > h
    @test_throws ArgumentError grid(1, R, HalfChebyshevGrid())      # M < 2
    @test_throws ArgumentError grid(M, 0.0, R, HalfChebyshevGrid()) # four-argument API not supported
    @test_throws ArgumentError grid(M, -R, HalfChebyshevGrid())     # R < 0
    @test_throws ArgumentError MappedGrid(0.0)                     # α = 0
    @test_throws ArgumentError MappedGrid(1.5)                     # α > 1
    @test_throws ArgumentError MappedGrid(0.5, 0)                  # order < 1
end
