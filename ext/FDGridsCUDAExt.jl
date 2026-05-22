module FDGridsCUDAExt

using CUDA,
      LinearAlgebra
using CUDA: i32

using FDGrids
using FDGrids: DiffMatrix

# TODO: test this for both forward and adjoint operation

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Move a DiffMatrix to the GPU.
CUDA.cu(d::DiffMatrix{T, WIDTH, OPTIMISE}) where {T, WIDTH, OPTIMISE} =
    DiffMatrix{T, WIDTH, OPTIMISE}(CUDA.cu(Array(d.coeffs)))
CUDA.cu(d::AdjointDiffMatrix) =
    AdjointDiffMatrix(CUDA.cu(d.parent), CUDA.cu(d.coeffs))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GPU kernel using cartesian indexing.
#
# One thread per logical output element. The flat thread index is decomposed
# into N-dimensional cartesian indices using the logical array size. The
# stencil is applied by varying the index along DIM — exactly like the CPU
# _make_ref approach, just with one thread per element instead of nested loops.
#
# Because indexing goes through getindex/setindex!, this is generic and works
# for any array type, including HaloArrays where those methods apply the halo
# offset transparently.
#
# N, DIM, WIDTH are Val parameters so the compiler sees them as constants
# and unrolls the index decomposition and stencil accumulation.
@generated function _gpu_kernel!(y, coeffs, x,
                                  sz::NTuple{N, Int32},
                                  ::Val{DIM},
                                  ::Val{WIDTH}) where {N, DIM, WIDTH}
    hWIDTH = WIDTH >> 1

    # Build index decomposition: i_1, i_2, ... i_N from flat 0-based idx
    decomp = quote
        rem = idx
    end
    for d in 1:N
        push!(decomp.args, :($(Symbol(:i_, d)) = rem % sz[$d] + 1))
        push!(decomp.args, :(rem = rem ÷ sz[$d]))
    end

    # Build stencil accumulation, unrolled over WIDTH taps.
    # For tap p, replace i_DIM with j0+p-1, keep all other indices.
    stencil = quote
        M      = sz[$DIM]
        hWIDTH = $hWIDTH
        j0 = $(Symbol(:i_, DIM)) ≤ hWIDTH     ? 1i32              :
             $(Symbol(:i_, DIM)) > M - hWIDTH ? M - $WIDTH + 1i32 :
             $(Symbol(:i_, DIM)) - hWIDTH
        s = zero(eltype(y))
    end
    for p in 1:WIDTH
        # build index tuple for this tap: (i_1, ..., j0+p-1, ..., i_N)
        idx_expr = Expr(:tuple, ntuple(d -> d == DIM ?
            :(j0 + $(p - 1)) :
            Symbol(:i_, d), N)...)
        push!(stencil.args, :(s += coeffs[$p, $(Symbol(:i_, DIM))] * x[$idx_expr...]))
    end

    # Write result
    out_idx = Expr(:tuple, ntuple(d -> Symbol(:i_, d), N)...)

    return quote
        idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x - 1i32
        idx ≥ prod(sz) && return nothing

        $decomp
        $stencil

        y[$out_idx...] = s

        return nothing
    end
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       A::DiffMatrix{T, WIDTH, OPTIMISE, <:CuArray},
                                       x::AbstractArray{S, N},
                                        ::Val{DIM}=Val(1);
                                nthreads::TH      =nothing) where {T, S, N, WIDTH, OPTIMISE, DIM, TH<:Union{Nothing, Int}}
    N   in 1:4 || throw(ArgumentError("N must be in 1:4"))
    DIM in 1:N || throw(ArgumentError("DIM must be in 1:N"))

    nthread_expr = if TH <: Nothing
        :(optimal_threads(y, A, x, sz, ::Val($DIM), Val($WIDTH); max_threads=prod(sz)))
    else
        quote
            _nthreads = nthreads
            _blocks   = cld(prod(sz), _nthreads)
        end
    end

    quote
        size(x, $DIM) == size(y, $DIM) == size(A.coeffs, 2) ||
            throw(ArgumentError("inconsistent sizes"))

        sz = size(x)
        $nthread_expr

        @cuda threads=_nthreads blocks=_blocks _gpu_kernel!(
            y, A.coeffs, x, sz, Val(Int32($DIM)), Val(Int32($WIDTH))
        )

        return y
    end
end

function optimal_threads(y, A, x, sz, ::Val{DIM}, ::Val{WIDTH}; max_threads=nothing) where {DIM, WIDTH}
    k = @cuda launch=false kernel!(
        y, A.coeffs, x, sz, Val(DIM), Val(WIDTH)
    )
    config = launch_configuration(k.fun)
    return isnothing(max_threads) ? config.threads : min(config.threads, prod(sz))
end

end
