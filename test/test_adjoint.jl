# ================================================================================
# Tests for AdjointDiffMatrix
# Included from runtests.jl after `using FDGrids`.
# ================================================================================

@testset "test AdjointDiffMatrix type                    " begin
    M     = 32
    width = 5
    xs, _ = grid(M, -1, 1, MappedGrid(0.5))
    D     = DiffMatrix(xs, width, 1)
    At    = adjoint(D)

    # adjoint returns the right type
    @test At isa AdjointDiffMatrix{Float64, width}

    # size is preserved
    @test size(At) == size(D)

    # double adjoint returns the parent (no copy)
    @test adjoint(At) === D

    # parent reference is the same object
    @test At.parent === D

    # getindex is the transpose of the forward matrix
    Df = full(D)
    for i in 1:M, j in 1:M
        @test At[i, j] == Df[j, i]
    end

    # setindex! is not supported
    @test_throws ArgumentError (At[1, 1] = 0.0)

    # full gives the transpose of the forward full matrix
    @test full(At) ≈ transpose(Df)
end

@testset "test adjoint corner cases                      " begin
    for width in (3, 5, 7)
        # exactly 2*WIDTH points — no body, should throw
        xs_bad, _ = grid(2 * width, -1, 1, MappedGrid(0.5))
        D_bad     = DiffMatrix(xs_bad, width, 1)
        @test_throws ArgumentError adjoint(D_bad)

        # one below threshold, should also throw
        xs_bad2, _ = grid(2 * width - 1, -1, 1, MappedGrid(0.5))
        D_bad2     = DiffMatrix(xs_bad2, width, 1)
        @test_throws ArgumentError adjoint(D_bad2)

        # one above threshold: succeeds and returns AdjointDiffMatrix
        xs_ok, _ = grid(2 * width + 1, -1, 1, MappedGrid(0.5))
        D_ok     = DiffMatrix(xs_ok, width, 1)
        @test adjoint(D_ok) isa AdjointDiffMatrix{Float64, width}
    end
end

@testset "test adjoint identity                          " begin
    # verify (v, D w) = (D* v, w) for various orders, widths, and grids
    M = 1024
    for ORDER in (1, 2), WIDTH in (3, 5, 7),
            xs in (range(-1, stop=1, length=M), grid(M, -1, 1, MappedGrid(0.5))[1])
        D = DiffMatrix(xs, WIDTH, ORDER)
        v = randn(M)
        w = randn(M)
        a = v' * mul!(zeros(M), D, w)
        b = mul!(zeros(M), adjoint(D), v)' * w
        @test a ≈ b
    end
end

@testset "test transpose N-D                             " begin
    # verify mul!(y, adjoint(D), x, Val(DIM)) against Df' applied slice-by-slice
    M     = 32
    OTHER = 3

    @testset "N=$N DIM=$DIM width=$width" for N in 1:4, DIM in 1:N, width in (3, 5, 7)
        M > 2 * width || continue

        xs, _ = grid(M, -1, 1, MappedGrid(0.5))
        D     = DiffMatrix(xs, width, 1)
        Df    = full(D)
        shape = ntuple(d -> d == DIM ? M : OTHER, N)

        x            = randn(shape...)
        y_diffmatrix = similar(x)
        y_full       = similar(x)

        mul!(y_diffmatrix, adjoint(D), x, Val(DIM))

        # reference: permute DIM to first axis, reshape to (M, :), apply Df',
        # result writes back through the view — no copies
        perm  = (DIM, setdiff(1:N, DIM)...)
        x_p   = PermutedDimsArray(x,      perm)
        y_p   = PermutedDimsArray(y_full, perm)
        other = prod(size(x_p)[2:end])
        mul!(reshape(y_p, M, other), Df', reshape(x_p, M, other))

        @test y_diffmatrix ≈ y_full
    end
end

@testset "test weighted adjoint                          " begin
    # verify weighted_adjoint(D, w) agrees with W⁻¹ D* W applied explicitly
    for M in (32, 64), width in (3, 5, 7)
        M > 2 * width || continue

        xs, _ = grid(M, -1, 1, MappedGrid(0.5))
        D     = DiffMatrix(xs, width, 1)
        w     = 1.0 .+ rand(M)   # strictly positive weights

        Dp = adjoint(D, w)

        # Dp should be an AdjointDiffMatrix
        @test Dp isa AdjointDiffMatrix{Float64, width}
        @test adjoint(Dp) === D

        # action on a random vector: Dp*x ≈ (1/w) .* (full(D)' * (w .* x))
        x     = randn(M)
        y_Dp  = similar(x)
        mul!(y_Dp, Dp, x)

        y_ref = (1 ./ w) .* (full(D)' * (w .* x))

        @test y_Dp ≈ y_ref

        # scalar indexing and dense expansion should reflect the weighted
        # adjoint coefficients, not merely the unweighted transpose.
        Dp_full_ref = Diagonal(1 ./ w) * full(D)' * Diagonal(w)
        @test full(Dp) ≈ Dp_full_ref
        for i in 1:M, j in 1:M
            @test Dp[i, j] ≈ Dp_full_ref[i, j]
        end

        # unweighted adjoint is the w=1 special case
        At   = adjoint(D)
        Dp1  = adjoint(D, ones(M))
        y_At  = similar(x)
        y_Dp1 = similar(x)
        mul!(y_At,  At,  x)
        mul!(y_Dp1, Dp1, x)
        @test y_At ≈ y_Dp1

        # weighted adjoint throws when w has the wrong length
        @test_throws ArgumentError adjoint(D, ones(M + 1))

        # weighted adjoint throws when size ≤ 2*WIDTH (no body)
        xs_bad, _ = grid(2 * width, -1, 1, MappedGrid(0.5))
        D_bad     = DiffMatrix(xs_bad, width, 1)
        @test_throws ArgumentError adjoint(D_bad, ones(2 * width))
    end
end
