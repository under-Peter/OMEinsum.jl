struct EinArray{T, N, NI, TT<:NTuple{NI,AbstractArray{T, M} where M}, LT<:NTuple{NI,Any}, CT<:CartesianIndices{N}} <: AbstractArray{T, N}
    xs::TT
    locs_xs::LT
    size::NTuple{N, Int}
    CIS::CT
end

@generated function EinArray(::EinCode{ixs, iy}, xs::NTuple{NI,AbstractArray{T, M} where M}, size_dict) where {T, NI, ixs, iy}
    inner_indices, outer_indices, locs_xs, locs_y = OMEinsum.indices_and_locs(ixs, iy)

    quote
        # find size for each leg
        outer_sizes = getindex.(Ref(size_dict), $outer_indices)
        inner_sizes = getindex.(Ref(size_dict), $inner_indices)

        # cartesian indices for outer and inner legs
        outer_ci = CartesianIndices((outer_sizes...,))
        inner_ci = CartesianIndices((inner_sizes...,))
        CIS = CartesianIndices((outer_ci.indices..., inner_ci.indices...))

        EinArray(xs, $locs_xs, (outer_sizes..., inner_sizes...), CIS)
    end
end

Base.size(A::EinArray) = A.size
Base.getindex(A::EinArray{T}, ind) where {T} = map_prod(T, A.xs, ind, A.locs_xs)
Base.getindex(A::EinArray{T}, inds::Int...) where {T} = map_prod(T, A.xs, inds, A.locs_xs)
CUDAnative.cudaconvert(A::EinArray) = EinArray(cudaconvert.(A.xs), A.locs_xs, A.size, A.CIS)
CuArrays.cu(A::EinArray) = EinArray(cu.(A.xs), A.locs_xs, A.size, A.CIS)

@inline function GPUArrays.thread_blocks_heuristic(x::Int, y::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_y = min(max_threads ÷ threads_x, y)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (x, y) ./ threads)
    threads, blocks
end

using Test
@testset "EinArray" begin
    locs_xs = (EinIndexer((8,8), (1,2)), EinIndexer((8,8),(2,3)))
    ixs = ((1,2), (2,3))
    iy = (1,3)
    x1 = randn(8, 8)
    x2 = randn(8, 8)
    arr = EinArray(EinCode(ixs, iy), (x1, x2), OMEinsum.get_size_dict(ixs, (x1, x2)))
    # outer first, then inner
    @test arr[CartesianIndex((3,4,2))] == x1[4,3]*x2[3,2]
    @test arr[3,4,2] == x1[4,3]*x2[3,2]
end
