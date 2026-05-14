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

end
