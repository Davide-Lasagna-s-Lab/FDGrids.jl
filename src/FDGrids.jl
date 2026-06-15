module FDGrids

import LinearAlgebra

include("utils.jl")
include("symmetry.jl")
include("diffmatrix.jl")
include("adjoint.jl")
include("linalg.jl")
include("matmul.jl")
include("grids.jl")

include("deprecated/deprecated.jl")

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
       basis_vector,
       symmetry,
       symmetry_left,
       symmetry_right,
       centre,
       Symmetry,
       NoSymmetry,
       EvenSymmetry,
       OddSymmetry

# required dummy definitions to make extension methods available
function optimal_forward_threads end
function optimal_adjoint_threads end

end
