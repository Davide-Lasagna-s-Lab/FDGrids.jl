export quadweights, _quadweights

# Compute quadrature weights for mesh points xs for a 
# composite rule using polynomials of order `order`.
function quadweights(xs::AbstractVector, order::Int)
    # check data is increasing
    # issorted(xs) || throw(ArgumentError("input not sorted"))
    issorted(xs) ? (xs_sorted = xs) : (xs_sorted = reverse(xs))

    # number of points
    N = length(xs_sorted)

    # partition xs so that we have at at least order+1 points
    w = zeros(length(xs_sorted))

    # ii and ie and the initial and final indices 
    ii, ie = 1, 1
    while ie < N
        ie = ii + order
        # if the next interval is smaller, just go till the end
        ie = N - ie < order ? N : ie
        rng = ii:ie
        w[rng] += _quadweights(xs_sorted[rng])
        ii = ie
    end

    # return to original order if originally reversed
    issorted(xs) ? nothing : reverse!(w)

    return w
end

# http://www2.math.umd.edu/~dlevy/classes/amsc466/lecture-notes/integration-chap.pdf
function _quadweights(xs::AbstractVector)
    # number of points
    N = length(xs)

    # find integral of polynomial of degree d from xs[1] to xs[N]
    b = [(xs[end]^(d+1) - xs[1]^(d+1))/(d+1) for d = 0:N-1]

    # evaluate polynomial up to degree d on points
    A = [xs[i]^d for d=0:N-1, i = 1:N]

    # return weights
    return A\b
end
