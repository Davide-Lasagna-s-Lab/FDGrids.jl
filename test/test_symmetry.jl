function _active_symmetry_rows(xs, width::Int, side::Symbol)
    HWIDTH = width >> 1
    side === :left  && return 1:HWIDTH
    side === :right && return (length(xs) - HWIDTH + 1):length(xs)
    throw(ArgumentError("side must be :left or :right"))
end

_symmetry_tuple(::Val{:left},  symmetry) = (symmetry, NoSymmetry())
_symmetry_tuple(::Val{:right}, symmetry) = (NoSymmetry(), symmetry)
_symmetry_tuple(side::Symbol, symmetry) = _symmetry_tuple(Val(side), symmetry)

function _check_symmetry_derivatives(xs, side::Symbol, symmetry, f, df, ddf;
                                     widths = (5, 7), atol = 1e-10)
    u = f.(xs)
    for width in widths
        rows = _active_symmetry_rows(xs, width, side)

        for (order, exact) in ((1, df), (2, ddf))
            D = DiffMatrix(xs, width, order; symmetry = _symmetry_tuple(side, symmetry))
            y = similar(u)
            mul!(y, D, u)
            @test y[rows] ≈ exact.(xs[rows]) atol = atol
        end
    end
end

@testset "symmetry metadata                         " begin
    xs = grid(10, -1, 1, UniformGrid()).xs

    D = DiffMatrix(xs, 5, 1)
    @test symmetry(D) == FDGrids.NO_SYMMETRY
    @test symmetry_left(D)  == NoSymmetry()
    @test symmetry_right(D) == NoSymmetry()
    @test symmetry_centre(D) == (nothing, nothing)

    sym = (OddSymmetry(first(xs)), EvenSymmetry(last(xs)))
    Ds = DiffMatrix(xs, 5, 1; symmetry = sym)
    @test symmetry(Ds) == sym
    @test symmetry_left(Ds) == sym[1]
    @test symmetry_right(Ds) == sym[2]
    @test symmetry_centre(Ds) == (first(xs), last(xs))
    @test symmetry(copy(Ds)) == sym

    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (EvenSymmetry(), NoSymmetry()))
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (NoSymmetry(), EvenSymmetry()))
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (OddSymmetry(), NoSymmetry()))
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (NoSymmetry(), OddSymmetry()))
    @test_throws MethodError EvenSymmetry("0")

    for bad in ((:bad, NoSymmetry()),
                (NoSymmetry(), :bad),
                (EvenSymmetry(first(xs)),),
                [EvenSymmetry(first(xs)), NoSymmetry()],
                EvenSymmetry(first(xs)))
        @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = bad)
    end
    @test_throws ArgumentError FDGrids.validate_symmetry((EvenSymmetry(first(xs)), 1))

    for width in 3:2:9
        D0 = DiffMatrix(xs, width, 1)
        Dn = DiffMatrix(xs, width, 1; symmetry = FDGrids.NO_SYMMETRY)
        @test FDGrids.full(D0) == FDGrids.full(Dn)
        @test D0.coeffs == Dn.coeffs
    end
end

@testset "mirror stencil locality                   " begin
    xs     = collect(range(0.0, 1.0; length = 21))
    N      = length(xs)
    width  = 5
    HWIDTH = width >> 1
    M0     = Matrix(DiffMatrix(xs, width, 1))
    changed(M) = [i for i in 1:N if @views M[i, :] != M0[i, :]]

    @test Matrix(DiffMatrix(xs, width, 1; symmetry = FDGrids.NO_SYMMETRY)) == M0

    Ml = Matrix(DiffMatrix(xs, width, 1; symmetry = (EvenSymmetry(first(xs)), NoSymmetry())))
    @test changed(Ml) ⊆ collect(1:HWIDTH)
    @test @views Ml[HWIDTH+1:end, :] == M0[HWIDTH+1:end, :]
    @test @views Ml[1:HWIDTH, :] != M0[1:HWIDTH, :]

    Mr = Matrix(DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), EvenSymmetry(last(xs)))))
    @test changed(Mr) ⊆ collect(N-HWIDTH+1:N)
    @test @views Mr[1:N-HWIDTH, :] == M0[1:N-HWIDTH, :]
    @test @views Mr[N-HWIDTH+1:N, :] != M0[N-HWIDTH+1:N, :]

    Mb = Matrix(DiffMatrix(xs, width, 1; symmetry = (EvenSymmetry(first(xs)), EvenSymmetry(last(xs)))))
    @test changed(Mb) ⊆ vcat(1:HWIDTH, N-HWIDTH+1:N)
    @test @views Mb[HWIDTH+1:N-HWIDTH, :] == M0[HWIDTH+1:N-HWIDTH, :]

    xs_off = collect(range(0.1, 1.0; length = 20))
    Moff0  = Matrix(DiffMatrix(xs_off, width, 1))
    Moff   = Matrix(DiffMatrix(xs_off, width, 1; symmetry = (EvenSymmetry(0.0), NoSymmetry())))
    @test all(isfinite, Moff)
    @test @views Moff[1:HWIDTH, :] != Moff0[1:HWIDTH, :]
    @test @views Moff[HWIDTH+1:end, :] == Moff0[HWIDTH+1:end, :]

    @test_throws ArgumentError DiffMatrix(xs, width, 1; symmetry = (EvenSymmetry(0.5), NoSymmetry()))
    @test_throws ArgumentError DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), OddSymmetry(0.5)))
end

@testset "functional symmetry verification          " begin
    # Behavioral verification: when the input function really has the declared
    # symmetry, every boundary row rewritten by symmetry should recover the
    # analytic first and second derivatives.
    cases = (
        (:left,  grid(24, -1, 1, GaussLobattoGrid()).xs,  :first),
        (:right, grid(24, -1, 1, GaussLobattoGrid()).xs,  :last),
        (:left,  collect(range(0.1, 1.0; length = 24)), 0.0),
        (:right, collect(range(-1.0, -0.1; length = 24)), 0.0),
    )

    for (side, xs, c_spec) in cases
        c = c_spec === :first ? first(xs) :
            c_spec === :last  ? last(xs)  : c_spec

        _check_symmetry_derivatives(xs, side, OddSymmetry(c),
            x -> (x - c)^3,
            x -> 3 * (x - c)^2,
            x -> 6 * (x - c))

        _check_symmetry_derivatives(xs, side, EvenSymmetry(c),
            x -> (x - c)^2,
            x -> 2 * (x - c),
            x -> 2 + zero(x))
    end

    xs = grid(8, -1, 1, GaussLobattoGrid()).xs
    u  = cos.(π .* xs)

    Dwrong = DiffMatrix(xs, 5, 2; symmetry = (OddSymmetry(-1.0), EvenSymmetry(1.0)))
    Dright = DiffMatrix(xs, 5, 2; symmetry = (EvenSymmetry(-1.0), EvenSymmetry(1.0)))
    ywrong = similar(u)
    yright = similar(u)
    mul!(ywrong, Dwrong, u)
    mul!(yright, Dright, u)

    @test !isapprox(ywrong[1], π^2; atol = 1e-2)
    @test  isapprox(yright[1], π^2; atol = 1e-2)
end
