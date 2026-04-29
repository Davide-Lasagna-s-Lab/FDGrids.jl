using InteractiveUtils
using BenchmarkTools
using LinearAlgebra
using FDGrids

M     = 300
xs    = gridpoints(M, -1, 1, 0.5)
width = 3

D = DiffMatrix(xs, width, 2)
D[1,   :] .= [1, zeros(M-1)...]
D[end, :] .= [zeros(M-1)..., 1]

b = randn(M)

# ── 1. LAPACK banded LU with partial pivoting ─────────────────────────────────
# lu(D) converts D to LAPACK's banded storage and calls dgbtrf! (with pivoting).
# ldiv! then calls dgbtrs!.
# Dispatch: ldiv!(::DiffMatrixLU, ::Vector)
luD_lapack = lu(D)
@btime ldiv!($luD_lapack, bc) setup=(bc = copy($b))

# ── 2. Specialised @generated tril/triu solve on DiffMatrix ──────────────────
# lu!(D) factorises in-place using _banded_lu!, overwriting D.coeffs directly.
# ldiv!(D, b) dispatches to the @generated _banded_tril_solve! / _banded_triu_solve!
# which unroll the inner loop at compile time via @nexprs and read directly from
# D.coeffs — no index translation, no pivoting.
# Dispatch: ldiv!(::DiffMatrix{T,WIDTH,OPTIMISE}, ::Vector)
luD_spec = lu!(copy(D))
@btime LinearAlgebra.ldiv!(DC_, bc) setup=(DC_ = copy($luD_spec); bc = copy($b))

# ── 3. Generic scalar banded solve (reference) ───────────────────────────────
# Same factorised matrix as above, but dispatched through the generic
# _banded_tril_solve!(::AbstractMatrix, ::Vector, p) / _banded_triu_solve!(... q)
# which use plain scalar loops with no unrolling and generic getindex on the matrix.
# Dispatch: ldiv!(::AbstractMatrix, ::Vector, p::Int, q::Int)
luD_gen = lu!(copy(D))
@btime LinearAlgebra.ldiv!(DC_, bc, $(width-1), $(width-1)) setup=(DC_ = copy($luD_gen); bc = copy($b))
