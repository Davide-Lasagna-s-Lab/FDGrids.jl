using LinearAlgebra
using FDGrids
using HaloArrays
using MPI
using Test

include("test_utils.jl")
include("test_diffmatrix.jl")
include("test_adjoint.jl")
include("test_linalg.jl")
include("test_matmul.jl")
include("test_decomposed.jl")
include("test_grids.jl")
include("test_deprecated.jl")
