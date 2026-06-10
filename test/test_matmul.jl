@testset "matvec/matmat product                     " begin
    for width = (3, 5, 7)
        D = DiffMatrix(gridpoints(50, -1, 1), width, 1)
        Df = FDGrids.full(D)

        u = rand(50)
        @test norm(Df * u .- D * u)/50 < 1e-14

        u = rand(50, 2)
        @test norm(Df * u .- D * u)/50 < 1e-14
    end
end

@testset "mul! accumulation mode                    " begin
    xs = gridpoints(12, -1, 1)
    D = DiffMatrix(xs, 5, 1)
    Dt = adjoint(D)

    u = rand(12, 3)

    for A in (D, Dt)
        reference = similar(u)
        mul!(reference, A, u, Val(1))

        base = rand(12, 3)
        out = copy(base)
        mul!(out, A, u, Val(1), Val(true))
        @test out ≈ base .+ reference

        # The default remains overwrite semantics.
        out = copy(base)
        mul!(out, A, u, Val(1))
        @test out ≈ reference
    end
end

@testset "test diffmatrix at point                  " begin
    # number of points
    for M in (10, 20, 30)
        for width in (3, 5, 7)
            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D = DiffMatrix(xs, width, 1)

            # arrange to 3D array
            fs = exp.(1.0.*xs)

            for i = 1:M
                @test mul!(similar(fs), D, fs)[i] == mul!(D, fs, i)
            end
        end
    end
end

@testset "test diffmatrix 1st/2nd order - vec       " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7),
                                                           (1.21, 3.22,  9.85),
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs = exp.(1.0.*xs)

            # exact first derivative
            d1fs_EX = copy(fs)

            # exact second derivative
            d2fs_EX = copy(fs)

            # compute finite difference approximation along the first direction
            d1fs_FD      = similar(fs)
            d2fs_FD      = similar(fs)
            mul!(d1fs_FD, D1, fs)
            mul!(d2fs_FD, D2, fs)

            # the relative error should scale like M^{-o} where o = width-1
            v1 = maximum(abs.(d1fs_EX - d1fs_FD))/maximum(abs.(d1fs_EX))*M^(width-1)
            @test v1 < v1_max

            # for the second derivative we have the same order in the domain center
            i = M >> 1
            v2 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-1)
            @test v2 < v2_center_max

            # and one order less at the boundary
            i = 1
            v3 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-2)
            @test v3 < v2_bndr_max
        end
    end
end

@testset "test diffmatrix 1st/2nd order - mat       " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7),
                                                           (1.21, 3.22,  9.85),
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs       = zeros(M, 2)
            fs[:, 1] = exp.(1.0 .* xs)
            fs[:, 2] = exp.(1.1 .* xs)

            # exact first derivative
            d1fs_EX       = copy(fs)
            d1fs_EX[:, 1] .*= 1.0
            d1fs_EX[:, 2] .*= 1.1

            # exact second derivative
            d2fs_EX       = copy(fs)
            d2fs_EX[:, 1] .*= 1.0^2
            d2fs_EX[:, 2] .*= 1.1^2

            # helper: view of arrays where the differentiated axis has length M
            # DIM = 1 -> (M,2) as-is
            # DIM = 2 -> (2,M) via permuted view (no copy)
            view_for_dim(A, DIM) = DIM == 1 ? A : PermutedDimsArray(A, (2, 1))

            # helper: index for center/boundary along the differentiated axis,
            # keeping the other index fixed to 1
            idx_on_axis(DIM, pos) = DIM == 1 ? CartesianIndex(pos, 1) : CartesianIndex(1, pos)

            for DIM in 1:2
                fs_view    = view_for_dim(fs, DIM)
                d1fs_EX_v  = view_for_dim(d1fs_EX, DIM)
                d2fs_EX_v  = view_for_dim(d2fs_EX, DIM)

                # compute finite difference approximation along the first direction
                d1fs_FD = similar(fs_view)
                d2fs_FD = similar(fs_view)
                mul!(d1fs_FD, D1, fs_view, Val(DIM))
                mul!(d2fs_FD, D2, fs_view, Val(DIM))

                # the relative error should scale like M^{-o} where o = width-1
                v1 = maximum(abs.(d1fs_EX_v - d1fs_FD)) / maximum(abs.(d1fs_EX_v)) * M^(width-1)
                @test v1 < v1_max

                # for the second derivative we have the same order in the domain center
                i  = M >> 1
                I  = idx_on_axis(DIM, i)
                v2 = abs(d2fs_EX_v[I] - d2fs_FD[I]) / abs(d2fs_EX_v[I]) * M^(width-1)
                @test v2 < v2_center_max

                # and one order less at the boundary
                i  = 1
                I  = idx_on_axis(DIM, i)
                v3 = abs(d2fs_EX_v[I] - d2fs_FD[I]) / abs(d2fs_EX_v[I]) * M^(width-2)
                @test v3 < v2_bndr_max
            end
        end
    end
end

@testset "test diffmatrix 1st/2nd order - cube      " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7),
                                                           (1.21, 3.22,  9.85),
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs          = zeros(M, 2, 2)
            fs[:, 1, 1] = exp.(1.0 .* xs)
            fs[:, 1, 2] = exp.(1.1 .* xs)
            fs[:, 2, 1] = exp.(1.2 .* xs)
            fs[:, 2, 2] = exp.(1.3 .* xs)

            # exact first derivative
            d1fs_EX          = copy(fs)
            d1fs_EX[:, 1, 1] .*= 1.0
            d1fs_EX[:, 1, 2] .*= 1.1
            d1fs_EX[:, 2, 1] .*= 1.2
            d1fs_EX[:, 2, 2] .*= 1.3

            # exact second derivative
            d2fs_EX          = copy(fs)
            d2fs_EX[:, 1, 1] .*= 1.0^2
            d2fs_EX[:, 1, 2] .*= 1.1^2
            d2fs_EX[:, 2, 1] .*= 1.2^2
            d2fs_EX[:, 2, 2] .*= 1.3^2

            # helper: view of arrays where the differentiated axis has length M
            # DIM = 1 -> (M,2,2) as-is
            # DIM = 2 -> (2,M,2) via permuted view (no copy)
            # DIM = 3 -> (2,2,M) via permuted view (no copy)
            view_for_dim(A, DIM) = DIM == 1 ? A :
                                   DIM == 2 ? PermutedDimsArray(A, (2, 1, 3)) :
                                              PermutedDimsArray(A, (2, 3, 1))

            # helper: index for center/boundary along the differentiated axis,
            # keeping the other indices fixed to 1
            idx_on_axis(DIM, pos) = DIM == 1 ? CartesianIndex(pos, 1, 1) :
                                  DIM == 2 ? CartesianIndex(1, pos, 1) :
                                             CartesianIndex(1, 1, pos)

            for DIM in 1:3
                fs_view    = view_for_dim(fs, DIM)
                d1fs_EX_v  = view_for_dim(d1fs_EX, DIM)
                d2fs_EX_v  = view_for_dim(d2fs_EX, DIM)

                # compute finite difference approximation along the first direction
                d1fs_FD = similar(fs_view)
                d2fs_FD = similar(fs_view)
                mul!(d1fs_FD, D1, fs_view, Val(DIM))
                mul!(d2fs_FD, D2, fs_view, Val(DIM))

                # the relative error should scale like M^{-o} where o = width-1
                v1 = maximum(abs.(d1fs_EX_v - d1fs_FD)) / maximum(abs.(d1fs_EX_v)) #*M^(width-1)
                @test v1 < v1_max

                # for the second derivative we have the same order in the domain center
                i  = M >> 1
                I  = idx_on_axis(DIM, i)
                v2 = abs(d2fs_EX_v[I] - d2fs_FD[I]) / abs(d2fs_EX_v[I]) * M^(width-1)
                @test v2 < v2_center_max

                # and one order less at the boundary
                i  = 1
                I  = idx_on_axis(DIM, i)
                v3 = abs(d2fs_EX_v[I] - d2fs_FD[I]) / abs(d2fs_EX_v[I]) * M^(width-2)
                @test v3 < v2_bndr_max
            end
        end
    end
end

@testset "test diffmatrix 1st/2nd order - hypercube " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7),
                                                           (1.21, 3.22,  9.85),
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # get grid
            xs = gridpoints(M, -1, 1, 0.5)

            # make grid from -1 to 1 using α = 0.5
            D1 = DiffMatrix(xs, width, 1)
            D2 = DiffMatrix(xs, width, 2)

            # arrange to 3D array
            fs             = zeros(M, 2, 2, 2)
            fs[:, 1, 1, 1] = exp.(1.0 .* xs)
            fs[:, 1, 2, 1] = exp.(1.1 .* xs)
            fs[:, 2, 1, 1] = exp.(1.2 .* xs)
            fs[:, 2, 2, 1] = exp.(1.3 .* xs)
            fs[:, 1, 1, 2] = exp.(1.4 .* xs)
            fs[:, 1, 2, 2] = exp.(1.5 .* xs)
            fs[:, 2, 1, 2] = exp.(1.6 .* xs)
            fs[:, 2, 2, 2] = exp.(1.7 .* xs)

            # exact first derivative
            d1fs_EX             = copy(fs)
            d1fs_EX[:, 1, 1, 1] .*= 1.0
            d1fs_EX[:, 1, 2, 1] .*= 1.1
            d1fs_EX[:, 2, 1, 1] .*= 1.2
            d1fs_EX[:, 2, 2, 1] .*= 1.3
            d1fs_EX[:, 1, 1, 2] .*= 1.4
            d1fs_EX[:, 1, 2, 2] .*= 1.5
            d1fs_EX[:, 2, 1, 2] .*= 1.6
            d1fs_EX[:, 2, 2, 2] .*= 1.7

            # exact second derivative
            d2fs_EX             = copy(fs)
            d2fs_EX[:, 1, 1, 1] .*= 1.0^2
            d2fs_EX[:, 1, 2, 1] .*= 1.1^2
            d2fs_EX[:, 2, 1, 1] .*= 1.2^2
            d2fs_EX[:, 2, 2, 1] .*= 1.3^2
            d2fs_EX[:, 1, 1, 2] .*= 1.4^2
            d2fs_EX[:, 1, 2, 2] .*= 1.5^2
            d2fs_EX[:, 2, 1, 2] .*= 1.6^2
            d2fs_EX[:, 2, 2, 2] .*= 1.7^2

            # helper: view of arrays where the differentiated axis has length M
            # DIM = 1 -> (M,2,2,2) as-is
            # DIM = 2 -> (2,M,2,2) via permuted view (no copy)
            # DIM = 3 -> (2,2,M,2) via permuted view (no copy)
            # DIM = 4 -> (2,2,2,M) via permuted view (no copy)
            view_for_dim(A, DIM) = DIM == 1 ? A :
                                   DIM == 2 ? PermutedDimsArray(A, (2, 1, 3, 4)) :
                                   DIM == 3 ? PermutedDimsArray(A, (2, 3, 1, 4)) :
                                              PermutedDimsArray(A, (2, 3, 4, 1))

            # helper: index for center/boundary along the differentiated axis,
            # keeping the other indices fixed to 1
            idx_on_axis(DIM, pos) = DIM == 1 ? CartesianIndex(pos, 1, 1, 1) :
                                    DIM == 2 ? CartesianIndex(1, pos, 1, 1) :
                                    DIM == 3 ? CartesianIndex(1, 1, pos, 1) :
                                               CartesianIndex(1, 1, 1, pos)

            for DIM in 1:4
                fs_view    = view_for_dim(fs, DIM)
                d1fs_EX_v  = view_for_dim(d1fs_EX, DIM)
                d2fs_EX_v  = view_for_dim(d2fs_EX, DIM)

                # compute finite difference approximation along the first direction
                d1fs_FD = similar(fs_view)
                d2fs_FD = similar(fs_view)
                mul!(d1fs_FD, D1, fs_view, Val(DIM))
                mul!(d2fs_FD, D2, fs_view, Val(DIM))

                # the relative error should scale like M^{-o} where o = width-1
                v1 = maximum(abs.(d1fs_EX_v - d1fs_FD)) / maximum(abs.(d1fs_EX_v)) #*M^(width-1)
                @test v1 < v1_max

                # for the second derivative we have the same order in the domain center
                i  = M >> 1
                I  = idx_on_axis(DIM, i)
                v2 = abs(d2fs_EX_v[I] - d2fs_FD[I]) / abs(d2fs_EX_v[I]) * M^(width-1)
                @test v2 < v2_center_max

                # and one order less at the boundary
                i  = 1
                I  = idx_on_axis(DIM, i)
                v3 = abs(d2fs_EX_v[I] - d2fs_FD[I]) / abs(d2fs_EX_v[I]) * M^(width-2)
                @test v3 < v2_bndr_max
            end
        end
    end
end
