export EinCode, EinIndexer, EinArray

"""
    EinCode{ixs, iy}

Einstein summation type notation, `ixs` are indice sets for input tensors
and `iy` is the index set for output tensor.
"""
struct EinCode{ixs, iy} end
EinCode(ixs::NTuple{N, NTuple{M, T} where M},iy::NTuple{<:Any,T}) where {N, T} = EinCode{ixs, iy}()

getixs(code::EinCode{ixs,iy}) where {ixs, iy} = ixs
getiy(code::EinCode{ixs,iy}) where {ixs, iy} = iy

"""
    EinIndexer{N, locs}

A structure for indexing `EinArray`s. `locs` is the index positions (among all indices).
"""
struct EinIndexer{N, locs}
    cumsize::NTuple{N, Int}
end

function einindexer(size::NTuple{N,Int}, locs::NTuple{N,Int}) where N
    N==0 && return EinIndexer{0,()}(())
    EinIndexer{N, locs}((1,TupleTools.cumprod(size[1:end-1])...))
end
getlocs(::EinIndexer{N,locs}) where {N,locs} = locs

subindex(indexer::EinIndexer, ind::CartesianIndex) = subindex(indexer, ind.I)
subindex(indexer::EinIndexer{0}, ind::NTuple{N0,Int}) where N0 = 1
@inline @generated function subindex(indexer::EinIndexer{N,locs}, ind::NTuple{N0,Int}) where {N,N0,locs}
    ex = Expr(:call, :+, map(i->i==1 ? :(ind[$(locs[i])]) : :((ind[$(locs[i])]-1) * indexer.cumsize[$i]), 1:N)...)
    :(@inbounds $ex)
end

"""
    EinArray{T, N, TT, LX, LY, ICT, OCT} <: AbstractArray{T, N}
    EinArray(::EinCode, xs, size_dict) -> EinArray

A virtual array as the expanded view of einstein summation array.
Indices are arranged as "inner indices (or reduced dimensions) first and then outer indices".

Type parameters are

    * T: element type,
    * N: array dimension,
    * NI: number of input tensors,
    * TT: type of "tuple of input arrays",
    * LX: type of "tuple of input indexers",
    * LX: type of output indexer,
    * ICT: typeof inner CartesianIndices,
    * OCT: typeof outer CartesianIndices,
"""
struct EinArray{T, N, NI, TT<:NTuple{NI,AbstractArray{T, M} where M}, LX, LY, ICT, OCT} <: AbstractArray{T, N}
    xs::TT
    x_indexers::LX
    y_indexer::LY
    size::NTuple{N, Int}
    ICIS::ICT
    OCIS::OCT
end

@generated function EinArray(::EinCode{ixs, iy}, xs::NTuple{NI,AbstractArray{T, M} where M}, size_dict) where {T, NI, ixs, iy}
    inner_indices, outer_indices, locs_xs, locs_y = indices_and_locs(ixs, iy)

    quote
        # find size for each leg
        outer_sizes = getindex.(Ref(size_dict), $outer_indices)
        inner_sizes = getindex.(Ref(size_dict), $inner_indices)

        # cartesian indices for outer and inner legs
        outer_ci = CartesianIndices((outer_sizes...,))
        inner_ci = CartesianIndices((inner_sizes...,))

        x_indexers = einindexer.(size.(xs), $locs_xs)
        y_size = getindex.(Ref(size_dict), iy)
        y_indexer = einindexer(y_size, $locs_y)

        EinArray(xs, x_indexers, y_indexer, (inner_sizes...,outer_sizes...), inner_ci, outer_ci)
    end
end

Base.size(A::EinArray) = A.size
Base.getindex(A::EinArray{T}, ind) where {T} = map_prod(A.xs, ind, A.x_indexers)
Base.getindex(A::EinArray{T}, inds::Int...) where {T} = map_prod(A.xs, inds, A.x_indexers)

@inline @generated function map_prod(xs::Tuple, ind, indexers::NTuple{N,Any}) where {N}
    ex = Expr(:call, :*, map(i->:(xs[$i][subindex(indexers[$i], ind)]), 1:N)...)
    :(@inbounds $ex)
end

# get inner indices, outer indices,
# locations of input indices in total indices
# and locations of output indices in outer indices.
function indices_and_locs(ixs, iy)
    # outer legs and inner legs
    outer_indices = tunique(iy)
    inner_indices = tsetdiff(TupleTools.vcat(ixs...), outer_indices)

    # for indexing tensors (leg binding)
    indices = (inner_indices...,outer_indices...)
    locs_xs = map(ixs) do ix
        map(i->findfirst(isequal(i), indices), ix)
    end
    locs_y = map(i->findfirst(isequal(i), outer_indices), iy)
    return inner_indices, outer_indices, locs_xs, locs_y
end
