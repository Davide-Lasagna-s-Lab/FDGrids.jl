# complete set of loop variables and array dimensions (up to array of dimension 4)
const __VARS__ = (:i, :j, :k, :l)
const __NS__ = (:N1, :N2, :N3, :N4)

# put `expr` into slot DIM, keep i, j, k elsewhere
function _make_ref(array, expr, DIM, N)
    inds = ntuple(d -> d == DIM ? expr : __VARS__[d], N)
    return Expr(:ref, array, inds...)
end

function _make_kernel(DIM, start_pos, WIDTH, N)
    return quote
        s = A.coeffs[1, $(__VARS__[DIM])] * $(_make_ref(:x, start_pos, DIM, N))
        Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
            s += A.coeffs[1 + p, $(__VARS__[DIM])] * $(_make_ref(:x, :( ($start_pos) + p), DIM, N))
        end
        $(_make_ref(:y, __VARS__[DIM], DIM, N)) = s
    end
end

@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       A::DiffMatrix{T, WIDTH},
                                       x::AbstractArray{S, N}, 
                                        ::Val{DIM} = Val(1)) where {T, S, N, WIDTH, DIM}
    # sanity checks
    N in 1:4 ||
        throw(ArgumentError("inconsistent array dimension"))

    DIM in 1:N ||
        throw(ArgumentError("inconsistent differentiation dimension"))
    
    # array size of the dimension we differentiate along
    q = __VARS__[DIM]
    Nq = __NS__[DIM]

    # stencil half width
    hWIDTH = WIDTH >> 1 

    # kernel expressions, starting at different locations
    head_kernel = _make_kernel(DIM, 1, WIDTH, N)
    body_kernel = _make_kernel(DIM, :($q - $hWIDTH), WIDTH, N)
    tail_kernel = _make_kernel(DIM, :($Nq - $WIDTH + 1), WIDTH, N)

    # ranges for the head, body and tail regions
    head_range = :(1:$hWIDTH)
    body_range = :((1 + $hWIDTH):($Nq - $hWIDTH))
    tail_range = :(($Nq - $hWIDTH + 1):$Nq)

    # N = 3
    block = 
        if N == 1
            quote
                for i = $head_range; $head_kernel; end
                for i = $body_range; $body_kernel; end
                for i = $tail_range; $tail_kernel; end
            end
        elseif N == 2
            if DIM == 1
                quote
                    for j = 1:N2
                        for i = $head_range; $head_kernel; end
                        for i = $body_range; $body_kernel; end
                        for i = $tail_range; $tail_kernel; end
                    end
                end
            else
                quote
                    for j = 1:$head_range
                        for i = 1:N1; $head_kernel; end
                    end
                    for j = 1:$body_range
                        for i = 1:N1; $body_kernel; end
                    end
                    for j = 1:$tail_range
                        for i = 1:N1; $tail_kernel; end
                    end
                end
            end
        elseif N == 3
            if DIM == 1
                quote
                    for k = 1:N3
                        for j = 1:N2
                            for i = $head_range; $head_kernel; end
                            for i = $body_range; $body_kernel; end
                            for i = $tail_range; $tail_kernel; end
                        end
                    end
                end
            elseif DIM == 2
                quote
                    for k = 1:N3
                        for j = $head_range; for i = 1:N1; $head_kernel; end; end
                        for j = $body_range; for i = 1:N1; $body_kernel; end; end
                        for j = $tail_range; for i = 1:N1; $tail_kernel; end; end
                    end
                end
            else # DIM == 3
                quote
                    for k = $head_range, j = 1:N2, i = 1:N1; $head_kernel; end
                    for k = $body_range, j = 1:N2, i = 1:N1; $body_kernel; end
                    for k = $tail_range, j = 1:N2, i = 1:N1; $tail_kernel; end
                end
            end
        else # N == 4
            if DIM == 1
                quote
                    for l = 1:N4, k = 1:N3, j = 1:N2
                        for i = $head_range; $head_kernel; end
                        for i = $body_range; $body_kernel; end
                        for i = $tail_range; $tail_kernel; end
                    end
                end
            elseif DIM == 2
                quote
                    for l = 1:N4, k = 1:N3
                        for j = $head_range, i = 1:N1; $head_kernel; end
                        for j = $body_range, i = 1:N1; $body_kernel; end
                        for j = $tail_range, i = 1:N1; $tail_kernel; end
                    end
                end
            elseif DIM == 3
                quote
                    for l = 1:N4
                        for k = $head_range, j = 1:N2, i = 1:N1; $head_kernel; end
                        for k = $body_range, j = 1:N2, i = 1:N1; $body_kernel; end
                        for k = $tail_range, j = 1:N2, i = 1:N1; $tail_kernel; end
                    end
                end
            else # DIM == 4
                quote
                    for l = $head_range, k = 1:N3, j = 1:N2, i = 1:N1; $head_kernel; end
                    for l = $body_range, k = 1:N3, j = 1:N2, i = 1:N1; $body_kernel; end
                    for l = $tail_range, k = 1:N3, j = 1:N2, i = 1:N1; $tail_kernel; end
                end
            end
        end

    # define N1, N2, ...
    Ni = [Symbol(:N, d) for d in 1:N]
    
    return quote
        # check sizes match along the dimension we differentiate along
        size(x, $DIM) == size(y, $DIM) == size(A.coeffs, 2) ||
            throw(ArgumentError("inconsistent inputs size"))

        # size of array as a tuple, e.g.
        # N1, N2, N3 = size(y)
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))

        @inbounds begin
            $block
        end
        
        return y
    end
end

# Find derivative of `x` at point `i` 
@generated function LinearAlgebra.mul!(A::DiffMatrix{T, WIDTH}, x::AbstractVector, i::Int) where {T, WIDTH}
    quote
        # size of vector
        N = length(x)

        # check size
        size(A, 2) == N || throw(DimensionMismatch())

        # index of the first element of the stencil
        left = clamp(i - $WIDTH>>1, 1, N - $WIDTH + 1)

        # expand expressions
        val = A.coeffs[1, i]*x[left]
        Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
            val += A.coeffs[1 + p, i]*x[left + p]
        end
        
        return val
    end
end

# differentiate x along direction 1 in a given range (special method required for halo arrays differentiation)
"""
    This function is primarily intended for use with halo arrays from the
    HaloArrays.jl (see https://github.com/Davide-Lasagna-s-Lab/HaloArrays.jl). 
    
    The goal is to be able to only compute the derivative at range of points
    in the domain `A` was generated for. This is equivalent to taking a slice
    of the DiffMatrix `A` and multiplying only the relevent coefficients with
    the input `x` to give the derivative within the desired `rng`.
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{S, 4},
                                       A::DiffMatrix{T, WIDTH},
                                       x::AbstractArray{S, 4},
                                     rng::AbstractRange) where {T, S, WIDTH}
    quote
        # size of array
        N1, N2, N3, N4 = size(y)

        @inbounds for j in 1:N2, k in 1:N3, l in 1:N4
            for (i, A_i) in enumerate(rng)
                # index of the first element of the stencil
                left = clamp(i - ($WIDTH >> 1), 1, length(rng))

                # compute derivatives
                s = A.coeffs[1, A_i]*x[left, j, k, l]
                Base.Cartesian.@nexprs $(WIDTH - 1) p -> begin
                    s += A.coeffs[1 + p, A_i]*x[left + p, j, k, l]
                end
                y[i, j, k, l] = s
            end
        end
        return y
    end
end
