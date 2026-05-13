@testset "utils                                    " begin
    e = basis_vector(3, 5)
    @test e == [0.0, 0.0, 1.0, 0.0, 0.0]

    e_int = basis_vector(2, 4, Int)
    @test e_int == [0, 1, 0, 0]
    @test eltype(e_int) === Int

    xs = [-1.0, 0.0, 1.0]
    weights = FDGrids.get_weights(0.0, xs, 2)
    @test weights[:, 1] ≈ [0.0, 1.0, 0.0]
    @test weights[:, 2] ≈ [-0.5, 0.0, 0.5]
    @test weights[:, 3] ≈ [1.0, -2.0, 1.0]
end
