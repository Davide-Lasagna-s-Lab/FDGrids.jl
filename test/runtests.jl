using LinearAlgebra
using FDGrids
using Test

# ----------------------- #
# test base functionality #
# ----------------------- #
if "skip_base" ∉ ARGS
    include("test_utils.jl")
    include("test_diffmatrix.jl")
    include("test_symmetry.jl")
    include("test_adjoint.jl")
    include("test_linalg.jl")
    include("test_matmul.jl")
    include("test_grids.jl")
end


# ------------------------------ #
# optionally test CUDA extension #
# ------------------------------ #
function cuda_available()
    try
        CUDA.functional()
    catch
        false
    end
end

if "gpu_ext" ∈ ARGS
    using CUDA
    if !cuda_available()
        @warn "Skipping GPU tests - CUDA not functional"
        @test_broken false
    else
        include("FDGridsCUDAExt/test_gpu.jl")
    end
end
