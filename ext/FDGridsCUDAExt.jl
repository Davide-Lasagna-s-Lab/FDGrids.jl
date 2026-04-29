module FDGridsCUDAExt

using CUDA
using FDGrids
using FDGrids: DiffMatrix
import LinearAlgebra

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Move a DiffMatrix to the GPU.
CUDA.cu(d::DiffMatrix{T, WIDTH, OPTIMISE, A}) where {T, WIDTH, OPTIMISE, A} =
    DiffMatrix{T, WIDTH, OPTIMISE, CuMatrix{T}}(CUDA.cu(Array(d.coeffs)))

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
                                  sz::NTuple{N, Int},
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
        j0 = $(Symbol(:i_, DIM)) ≤ hWIDTH      ? 1              :
             $(Symbol(:i_, DIM)) > M - hWIDTH   ? M - $WIDTH + 1  :
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
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
        idx ≥ prod(sz) && return nothing

        $decomp
        $stencil

        y[$out_idx...] = s

        return nothing
    end
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       A::DiffMatrix{T, WIDTH, OPTIMISE, <:CuMatrix},
                                       x::AbstractArray{S, N},
                                        ::Val{DIM} = Val(1);
                                        nthreads::Int=256) where {T, S, N, WIDTH, OPTIMISE, DIM}
    N   in 1:4 || throw(ArgumentError("N must be in 1:4"))
    DIM in 1:N || throw(ArgumentError("DIM must be in 1:N"))

    quote
        size(x, $DIM) == size(y, $DIM) == size(A.coeffs, 2) ||
            throw(ArgumentError("inconsistent sizes"))

        sz = size(x)
        @cuda threads=nthreads blocks=cld(prod(sz), nthreads) _gpu_kernel!(
            y, A.coeffs, x, sz, Val($DIM), Val($WIDTH))

        return y
    end
end

end # module
