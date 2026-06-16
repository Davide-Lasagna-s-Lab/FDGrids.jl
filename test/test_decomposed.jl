# ================================================================================
# Tests for the decomposed-domain LinearAlgebra.mul! interface
# ================================================================================
#
# These tests exercise two valid local-storage conventions:
#
#   1. A dense array whose ordinary axes include the halo rows.
#   2. A HaloArray whose ordinary axes describe owned rows and whose halo rows
#      are addressed with indices such as 0, -1, and n + 1.
#
# In both cases `global_idx` is the global row represented by local index `1`,
# and `local_rng` selects the local rows whose outputs must be computed exactly.

MPI.Initialized() || MPI.Init()

const DECOMPOSED_GLOBAL_SIZE = 48
const DECOMPOSED_LOCAL_SIZE  = 12
const DECOMPOSED_OTHER_SIZE  = 2

# Use a non-polynomial profile and vary the transverse indices so the comparison
# checks every fibre rather than repeatedly applying the same one-dimensional
# input.
function _decomposed_input(shape::NTuple{N, Int}, dim::Int) where {N}
    x = Array{Float64}(undef, shape)
    for I in CartesianIndices(x)
        transverse = sum(d == dim ? 0.0 : 0.03 * d * I[d] for d in 1:N)
        x[I] = sin(0.17 * I[dim]) + transverse
    end
    return x
end

_decomposed_shape(N::Int, dim::Int, n::Int) =
    ntuple(d -> d == dim ? n : DECOMPOSED_OTHER_SIZE, N)

function _decomposed_slabs(M::Int, n::Int)
    middle = ((M - n) ÷ 2 + 1):((M - n) ÷ 2 + n)
    return (lower=1:n, middle=middle, upper=(M - n + 1):M)
end

# A dense decomposed buffer stores valid halo rows directly in its ordinary
# axes. At a physical domain edge there is no out-of-domain halo storage.
function _test_dense_halos(op, x_global, y_reference, dim::Int, owned, nhalo::Int, add::Bool)
    M = size(x_global, dim)
    stored = max(1, first(owned) - nhalo):min(M, last(owned) + nhalo)
    local_rng = (first(owned) - first(stored) + 1):(last(owned) - first(stored) + 1)

    x_local = copy(selectdim(x_global, dim, stored))
    y_local = fill(11.0, size(x_local))
    base    = copy(y_local)

    mul!(y_local, op, x_local, Val(dim), first(stored), local_rng, Val(add))

    expected = add ? selectdim(base, dim, local_rng) .+
                     selectdim(y_reference, dim, owned) :
                     selectdim(y_reference, dim, owned)
    return isapprox(selectdim(y_local, dim, local_rng), expected)
end

# Copy all in-domain values needed by an owned slab into a HaloArray. HaloArray
# scalar indexing shifts logical halo indices into its dense parent storage.
function _fill_haloarray!(x_local::HaloArray, x_global, dim::Int, owned, nhalo::Int)
    N = ndims(x_local)
    ranges = ntuple(d -> d == dim ? ((1 - nhalo):(length(owned) + nhalo)) :
                                    axes(x_local, d), N)
    for I in CartesianIndices(ranges)
        global_dim_index = first(owned) + I[dim] - 1
        1 <= global_dim_index <= size(x_global, dim) || continue
        J = ntuple(d -> d == dim ? global_dim_index : I[d], N)
        x_local[Tuple(I)...] = x_global[J...]
    end
    return x_local
end

# A HaloArray exposes owned rows as 1:n and the ghost rows through logical
# indices outside that range. This is the convention used by NSEBaseMPIExt.
function _test_haloarray_halos(op, x_global, y_reference, dim::Int, owned, nhalo::Int, add::Bool)
    N = ndims(x_global)
    comm = MPI.Cart_create(MPI.COMM_SELF, ntuple(_ -> 1, N);
                           periodic=ntuple(_ -> false, N), reorder=false)
    local_shape = _decomposed_shape(N, dim, length(owned))
    halos       = ntuple(d -> d == dim ? nhalo : 0, N)
    try
        x_local = HaloArray{Float64}(comm, local_shape, halos)
        y_local = similar(x_local)

        _fill_haloarray!(x_local, x_global, dim, owned, nhalo)
        parent(y_local) .= 11.0
        base = copy(y_local)

        local_rng = 1:length(owned)
        mul!(y_local, op, x_local, Val(dim), first(owned), local_rng, Val(add))

        expected = add ? selectdim(base, dim, local_rng) .+
                         selectdim(y_reference, dim, owned) :
                         selectdim(y_reference, dim, owned)
        return isapprox(selectdim(y_local, dim, local_rng), expected)
    finally
        MPI.free(comm)
    end
end

@testset "decomposed mul! exactness" begin
    M = DECOMPOSED_GLOBAL_SIZE
    n = DECOMPOSED_LOCAL_SIZE
    dense_failures     = String[]
    haloarray_failures = String[]

    for N          in 1:4,
        dim        in 1:N,
        width      in (3, 5, 7),
        order      in (1, 2),
        is_adjoint in (false, true),
        add        in (false, true),
        slab       in keys(_decomposed_slabs(M, n))

        shape       = _decomposed_shape(N, dim, M)
        x_global    = _decomposed_input(shape, dim)
        D           = DiffMatrix(gridpoints(M, -1, 1, 0.5), width, order)
        op          = is_adjoint ? adjoint(D) : D
        y_reference = similar(x_global)
        mul!(y_reference, op, x_global, Val(dim))

        owned  = getproperty(_decomposed_slabs(M, n), slab)
        nhalo = width >> 1
        label  = "N=$N dim=$dim width=$width order=$order adjoint=$is_adjoint add=$add slab=$slab"

        _test_dense_halos(op, x_global, y_reference, dim, owned, nhalo, add) ||
            push!(dense_failures, label)
        _test_haloarray_halos(op, x_global, y_reference, dim, owned, nhalo, add) ||
            push!(haloarray_failures, label)
    end

    @testset "dense arrays with explicit halos" begin
        isempty(dense_failures) ||
            @info "dense decomposed failures" count=length(dense_failures) examples=first(dense_failures, min(6, length(dense_failures)))
        @test length(dense_failures) == 0
    end

    @testset "HaloArray logical halos" begin
        isempty(haloarray_failures) ||
            @info "HaloArray decomposed failures" count=length(haloarray_failures) examples=first(haloarray_failures, min(6, length(haloarray_failures)))
        @test length(haloarray_failures) == 0
    end
end
