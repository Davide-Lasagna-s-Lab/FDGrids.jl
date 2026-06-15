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
       ChebyshevGrid,
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

end
