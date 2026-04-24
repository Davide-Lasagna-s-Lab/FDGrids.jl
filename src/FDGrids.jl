module FDGrids

import LinearAlgebra

include("utils.jl")
include("diffmatrix.jl")
include("adjoint.jl")
include("linalg.jl")
include("matmul.jl")
include("grids.jl")
include("quadrature.jl")

export DiffMatrix,
       AdjointDiffMatrix,
       gridpoints,
       full,
       weighted_adjoint,
       banded_lu!,
       banded_tril_solve!,
       banded_triu_solve,
       basis_vector

end
