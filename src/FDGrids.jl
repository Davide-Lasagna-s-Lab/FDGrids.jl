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
       banded_lu!,
       banded_tril_solve!,
       banded_triu_solve,
       basis_vector

# required dummy definitions to make extension methods available
function optimal_forward_threads end
function optimal_adjoint_threads end

end
