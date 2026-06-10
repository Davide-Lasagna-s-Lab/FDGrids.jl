module FDGrids

import LinearAlgebra

include("utils.jl")
include("diffmatrix.jl")
include("adjoint.jl")
include("linalg.jl")
include("matmul.jl")
include("grids.jl")
include("deprecated.jl")

export DiffMatrix,
       DiffMatrixLU,
       AdjointDiffMatrix,
       AbstractGridDistribution,
       MappedGrid,
       UniformGrid,
       GaussLobattoGrid,
       grid,
       gridpoints,
       full,
       quadweights,
       _quadweights,
       BandedMatrixLU,
       basis_vector

# required dummy definitions to make extension methods available
function optimal_forward_threads end
function optimal_adjoint_threads end

end
