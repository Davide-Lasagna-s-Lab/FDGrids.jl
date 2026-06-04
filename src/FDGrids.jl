module FDGrids

import LinearAlgebra

include("utils.jl")
include("symmetry.jl")
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
       basis_vector,
       symmetry,
       symmetry_left,
       symmetry_right,
       symmetry_centre,
       symmetry_centre_left,
       symmetry_centre_right,
       Symmetry,
       NoSymmetry,
       Even,
       Odd,
       NO_SYMMETRY

end
