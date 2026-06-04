@testset "symmetry metadata                         " begin
    xs = gridpoints(10, -1, 1, 1)

    # 1. default is (NoSymmetry(), NoSymmetry())
    D = DiffMatrix(xs, 5, 1)
    @test symmetry(D) == (NoSymmetry(), NoSymmetry())
    @test symmetry(D) == FDGrids.NO_SYMMETRY
    @test symmetry_left(D)  == NoSymmetry()
    @test symmetry_right(D) == NoSymmetry()

    # 2. explicit values are stored and read back
    for sym in ((Even(), NoSymmetry()), (Odd(), Even()), (NoSymmetry(), Odd()), (Even(), Odd()))
        Ds = DiffMatrix(xs, 5, 1; symmetry = sym)
        @test symmetry(Ds)       == sym
        @test symmetry_left(Ds)  == sym[1]
        @test symmetry_right(Ds) == sym[2]
    end

    # 3. invalid input throws ArgumentError
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (:bad, NoSymmetry()))   # not a Symmetry
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (NoSymmetry(), :bad))    # not a Symmetry
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (Even(),))               # wrong arity
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = [Even(), NoSymmetry()])  # not a tuple
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = Even())                  # not a tuple
    @test_throws ArgumentError FDGrids.validate_symmetry((Even(), 1))                   # non-Symmetry element

    # a non-Real, non-nothing centre is rejected by the symmetry type itself
    @test_throws MethodError Even("0")

    # validate_symmetry returns the tuple unchanged for valid input
    @test FDGrids.validate_symmetry((Even(), Odd())) == (Even(), Odd())

    # 4. a fully NoSymmetry operator is identical to the plain constructor
    for width = 3:2:9
        D0 = DiffMatrix(xs, width, 1)
        Dn = DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), NoSymmetry()))
        @test FDGrids.full(D0) == FDGrids.full(Dn)
        @test D0.coeffs == Dn.coeffs

        # ... and the operator action is unchanged
        x  = cos.(xs)
        y0 = similar(x); yn = similar(x)
        mul!(y0, D0, x)
        mul!(yn, Dn, x)
        @test y0 ≈ yn
    end

    # copy preserves the metadata
    Dc = copy(DiffMatrix(xs, 5, 1; symmetry = (Odd(), Even())))
    @test symmetry(Dc) == (Odd(), Even())
end

@testset "symmetry centre metadata                  " begin
    # use a [0, 1] grid so that left centre 0.0 / right centre 1.0 are valid
    xs = collect(range(0.0, 1.0; length = 11))

    # default: NoSymmetry carries no centre
    D = DiffMatrix(xs, 5, 1)
    @test symmetry_centre(D) == (nothing, nothing)
    @test symmetry_centre_left(D)  === nothing
    @test symmetry_centre_right(D) === nothing

    # an unspecified centre (Even()/Odd()) also reads back as nothing
    Dd = DiffMatrix(xs, 5, 1; symmetry = (Even(), NoSymmetry()))
    @test symmetry_centre(Dd) == (nothing, nothing)

    # explicit left centre
    Dl = DiffMatrix(xs, 5, 1; symmetry = (Even(0.0), NoSymmetry()))
    @test symmetry_centre(Dl)       == (0.0, nothing)
    @test symmetry_centre_left(Dl)  == 0.0
    @test symmetry_centre_right(Dl) === nothing

    # both centres
    Db = DiffMatrix(xs, 5, 1; symmetry = (Even(0.0), Odd(1.0)))
    @test symmetry_centre(Db) == (0.0, 1.0)

    # copy preserves the metadata
    Dc = DiffMatrix(xs, 5, 1; symmetry = (Even(0.0), NoSymmetry()))
    @test symmetry(copy(Dc))        == symmetry(Dc)
    @test symmetry_centre(copy(Dc)) == symmetry_centre(Dc)
end

@testset "boundary-centred mirror stencils          " begin
    xs = collect(range(0.0, 1.0; length = 21))
    D0 = DiffMatrix(xs, 5, 1)

    # 1. no symmetry is identical to the plain constructor
    Dn = DiffMatrix(xs, 5, 1; symmetry = (NoSymmetry(), NoSymmetry()))
    @test Matrix(D0) ≈ Matrix(Dn)

    # 2. an active side changes the boundary coefficients
    De = DiffMatrix(xs, 5, 1; symmetry = (Even(), NoSymmetry()))
    @test !(Matrix(D0) ≈ Matrix(De))

    # 3. left even: derivative of an even function vanishes at the centre
    let u = xs .^ 2
        D  = DiffMatrix(xs, 5, 1; symmetry = (Even(0.0), NoSymmetry()))
        du = Matrix(D) * u
        @test du[1] ≈ 0 atol = 1e-10
    end

    # 4. left odd: derivative of an odd function is recovered at the centre
    let u = copy(xs)
        D  = DiffMatrix(xs, 5, 1; symmetry = (Odd(0.0), NoSymmetry()))
        du = Matrix(D) * u
        @test du[1] ≈ 1 atol = 1e-10
    end

    # 5. right even: derivative of an even function vanishes at the centre
    let u = (xs .- last(xs)) .^ 2
        D  = DiffMatrix(xs, 5, 1; symmetry = (NoSymmetry(), Even(last(xs))))
        du = Matrix(D) * u
        @test du[end] ≈ 0 atol = 1e-10
    end

    # 6. right odd: derivative of an odd function is recovered at the centre
    let u = xs .- last(xs)
        D  = DiffMatrix(xs, 5, 1; symmetry = (NoSymmetry(), Odd(last(xs))))
        du = Matrix(D) * u
        @test du[end] ≈ 1 atol = 1e-10
    end

    # 7. a centre off the grid still works and differs from the plain stencil
    let xs2 = collect(range(0.1, 1.0; length = 20))
        Dref = DiffMatrix(xs2, 5, 1)
        Doff = DiffMatrix(xs2, 5, 1; symmetry = (Even(0.0), NoSymmetry()))
        @test all(isfinite, Matrix(Doff))
        @test !(Matrix(Dref) ≈ Matrix(Doff))
    end

    # 8. a NoSymmetry side is left exactly as the plain one-sided stencil
    let HWIDTH = 5 >> 1
        Dright = DiffMatrix(xs, 5, 1; symmetry = (NoSymmetry(), Even(last(xs))))
        @test Matrix(Dright)[1:HWIDTH, :] == Matrix(D0)[1:HWIDTH, :]
    end

    # centre placed inside the grid is rejected for an active side
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (Even(0.5), NoSymmetry()))
    @test_throws ArgumentError DiffMatrix(xs, 5, 1; symmetry = (NoSymmetry(), Even(0.5)))
end

@testset "mirror stencil verification               " begin
    xs     = collect(range(0.0, 1.0; length = 21))
    N      = length(xs)
    width  = 5
    HWIDTH = width >> 1
    D0     = DiffMatrix(xs, width, 1)
    M0     = Matrix(D0)

    # indices of rows that differ from the plain operator
    changed(M) = [i for i in 1:N if @views M[i, :] != M0[i, :]]

    # 1. NoSymmetry is bit-identical to the plain operator
    @test Matrix(DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), NoSymmetry()))) == M0

    # 2. left symmetry only touches the left boundary rows (1:HWIDTH)
    Dl = DiffMatrix(xs, width, 1; symmetry = (Even(), NoSymmetry()))
    @test changed(Matrix(Dl)) ⊆ collect(1:HWIDTH)
    @test @views Matrix(Dl)[HWIDTH+1:end, :] == M0[HWIDTH+1:end, :]   # rest untouched
    @test @views Matrix(Dl)[1:HWIDTH, :]     != M0[1:HWIDTH, :]       # boundary changed

    # 3. right symmetry only touches the right boundary rows (N-HWIDTH+1:N)
    Dr = DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), Even()))
    @test changed(Matrix(Dr)) ⊆ collect(N-HWIDTH+1:N)
    @test @views Matrix(Dr)[1:N-HWIDTH, :]   == M0[1:N-HWIDTH, :]     # rest untouched
    @test @views Matrix(Dr)[N-HWIDTH+1:N, :] != M0[N-HWIDTH+1:N, :]   # boundary changed

    # 4. with both sides active, width=5 changes exactly 2 rows on each end
    Db = DiffMatrix(xs, width, 1; symmetry = (Even(0.0), Even(last(xs))))
    @test changed(Matrix(Db)) ⊆ vcat(1:HWIDTH, N-HWIDTH+1:N)
    @test @views Matrix(Db)[HWIDTH+1:N-HWIDTH, :] == M0[HWIDTH+1:N-HWIDTH, :]  # interior untouched

    # 5. left even: derivative of an even function vanishes at the centre
    let De = DiffMatrix(xs, width, 1; symmetry = (Even(0.0), NoSymmetry()))
        @test (Matrix(De) * (xs .^ 2))[1] ≈ 0 atol = 1e-10
    end

    # 6. left odd: derivative of an odd function recovers the slope
    let Do = DiffMatrix(xs, width, 1; symmetry = (Odd(0.0), NoSymmetry()))
        @test (Matrix(Do) * xs)[1] ≈ 1 atol = 1e-10
    end

    # 7. right even / odd behave the same way at the right centre
    let De = DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), Even(last(xs))))
        @test (Matrix(De) * ((xs .- last(xs)) .^ 2))[end] ≈ 0 atol = 1e-10
    end
    let Do = DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), Odd(last(xs))))
        @test (Matrix(Do) * (xs .- last(xs)))[end] ≈ 1 atol = 1e-10
    end

    # 8. centre off the grid (xs starts at 0.1, centre 0.0) runs and changes the left rows
    let xs2 = collect(range(0.1, 1.0; length = 20)), M0b = Matrix(DiffMatrix(xs2, width, 1))
        Doff = DiffMatrix(xs2, width, 1; symmetry = (Even(0.0), NoSymmetry()))
        Moff = Matrix(Doff)
        @test all(isfinite, Moff)
        @test @views Moff[1:HWIDTH, :]     != M0b[1:HWIDTH, :]            # left rows changed
        @test @views Moff[HWIDTH+1:end, :] == M0b[HWIDTH+1:end, :]        # rest untouched
    end

    # 9. an active side with its centre inside the grid throws
    @test_throws ArgumentError DiffMatrix(xs, width, 1; symmetry = (Even(0.5), NoSymmetry()))
    @test_throws ArgumentError DiffMatrix(xs, width, 1; symmetry = (NoSymmetry(), Odd(0.5)))
end
