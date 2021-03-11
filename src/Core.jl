export EinCode, EinIndexer, EinArray
export einarray

"""
    EinCode{ixs, iy}

Wrapper to `eincode`-specification that creates a callable object
to evaluate the `eincode` `ixs -> iy` where `ixs` are the index-labels
of the input-tensors and `iy` are the index-labels of the output

# example

```jldoctest; setup = :(using OMEinsum)
julia> a, b = rand(2,2), rand(2,2);

julia> EinCode((('i','j'),('j','k')),('i','k'))(a, b) ≈ a * b
true
```
"""
struct EinCode{ixs, iy} end
EinCode(ixs::NTuple{N, NTuple{M, T} where M},iy::NTuple{<:Any,T}) where {N, T} = EinCode{ixs, iy}()

getixs(code::EinCode{ixs,iy}) where {ixs, iy} = ixs
getiy(code::EinCode{ixs,iy}) where {ixs, iy} = iy

"""
    EinIndexer{locs,N}

A structure for indexing `EinArray`s. `locs` is the index positions (among all indices).
In the constructor, `size` is the size of target tensor,
"""
struct EinIndexer{locs,N}
    cumsize::NTuple{N, Int}
end

"""
    EinIndexer{locs}(size::Tuple)

Constructor for `EinIndexer` for an object of size `size` where `locs` are the
locations of relevant indices in a larger tuple.
"""
function EinIndexer{locs}(size::NTuple{N,Int}) where {N,locs}
    N==0 && return EinIndexer{(),0}(())
    EinIndexer{locs,N}((1,TupleTools.cumprod(size[1:end-1])...))
end

getlocs(::EinIndexer{locs,N}) where {N,locs} = locs

# get a subset of index
@inline subindex(indexer::EinIndexer, ind::CartesianIndex) = subindex(indexer, ind.I)
@inline subindex(indexer::EinIndexer{(),0}, ind::NTuple{N0,Int}) where N0 = 1
@inline @generated function subindex(indexer::EinIndexer{locs,N}, ind::NTuple{N0,Int}) where {N,N0,locs}
    ex = Expr(:call, :+, map(i->i==1 ? :(ind[$(locs[i])]) : :((ind[$(locs[i])]-1) * indexer.cumsize[$i]), 1:N)...)
    :(@inbounds $ex)
end

"""
    EinArray{T, N, TT, LX, LY, ICT, OCT} <: AbstractArray{T, N}

A struct to hold the intermediate result of an `einsum` where all index-labels
of both input and output are expanded to a rank-`N`-array whose values
are lazily calculated.
Indices are arranged as _inner indices_ (or reduced dimensions) first and _then outer indices_.

Type parameters are

    * `T`: element type,
    * `N`: array dimension,
    * `TT`: type of "tuple of input arrays",
    * `LX`: type of "tuple of input indexers",
    * `LX`: type of output indexer,
    * `ICT`: typeof inner CartesianIndices,
    * `OCT`: typeof outer CartesianIndices,
"""
struct EinArray{T, N, TT, LX, LY, ICT, OCT} <: AbstractArray{T, N}
    xs::TT
    x_indexers::LX
    y_indexer::LY
    size::NTuple{N, Int}
    ICIS::ICT
    OCIS::OCT
    function EinArray{T}(xs::TT, x_indexers::LX, y_indexer::LY, size::NTuple{N, Int}, ICIS::ICT, OCIS::OCT) where {T,N,TT,LX,LY,ICT,OCT}
        new{T,N,TT,LX,LY,ICT,OCT}(xs,x_indexers,y_indexer,size,ICIS,OCIS)
    end
end

"""
    einarray(::EinCode, xs, size_dict) -> EinArray

Constructor of `EinArray` from an `EinCode`, a tuple of tensors `xs` and a `size_dict`
of type `IndexSize` that assigns each index-label a size.
The returned `EinArray` holds an intermediate result of the `einsum` specified by the
`EinCode` with indices corresponding to all unique labels in the einsum.
Reduction over the (lazily calculated) dimensions that correspond to labels not present
in the output lead to the result of the einsum.

# example

```jldoctest; setup = :(using OMEinsum)
julia> using OMEinsum: get_size_dict

julia> a, b = rand(2,2), rand(2,2);

julia> sd = get_size_dict((('i','j'),('j','k')), (a, b));

julia> ea = OMEinsum.einarray(EinCode((('i','j'),('j','k')),('i','k')), (a,b), sd);

julia> dropdims(sum(ea, dims=1), dims=1) ≈ a * b
true
```
"""
@generated function einarray(::EinCode{ixs, iy}, xs::TT, size_dict) where {ixs, iy, NI, TT<:NTuple{NI,AbstractArray}}
    inner_indices, outer_indices, locs_xs, locs_y = indices_and_locs(ixs, iy)
    inner_indices = (inner_indices...,)
    outer_indices = (outer_indices...,)
    T = promote_type(eltype.(xs.parameters)...)
    xind_expr = Expr(:call, :tuple, map(i->:(EinIndexer{$(locs_xs[i])}(size(xs[$i]))),1:NI)...)

    quote
        # find size for each leg
        outer_sizes = getindex.(Ref(size_dict), $outer_indices)
        inner_sizes = getindex.(Ref(size_dict), $inner_indices)

        # cartesian indices for outer and inner legs
        outer_ci = CartesianIndices(outer_sizes)
        inner_ci = CartesianIndices(inner_sizes)

        x_indexers = $(xind_expr)
        y_size = getindex.(Ref(size_dict), iy)
        y_indexer = EinIndexer{$locs_y}(y_size)

        #EinArray{$T, $N, $TT, $LX, $LY, $ICT, $OCT}(xs, x_indexers, y_indexer, (inner_sizes...,outer_sizes...), inner_ci, outer_ci)
        EinArray{$T}(xs, x_indexers, y_indexer, (inner_sizes...,outer_sizes...), inner_ci, outer_ci)
    end
end

Base.size(A::EinArray) = A.size
@doc raw"
    getindex(A::EinArray, inds...)
return the lazily calculated entry of `A` at index `inds`.
"
@inline Base.getindex(A::EinArray{T}, ind) where {T} = map_prod(A.xs, ind, A.x_indexers)
@inline Base.getindex(A::EinArray{T}, inds::Int...) where {T} = map_prod(A.xs, inds, A.x_indexers)

# get one element from each tensor, and multiple them
@doc raw"
    map_prod(xs, ind, indexers)

calculate the value of an `EinArray` with `EinIndexer`s `indexers` at location `ind`.
"
@inline @generated function map_prod(xs::Tuple, ind, indexers::NTuple{N,Any}) where {N}
    ex = Expr(:call, :*, map(i->:(xs[$i][subindex(indexers[$i], ind)]), 1:N)...)
    :(@inbounds $ex)
end

@doc raw"
    indices_and_locs(ixs,iy)
given the index-labels of input and output of an `einsum`, return
(in the same order):
- a tuple of the distinct index-labels of the output `iy`
- a tuple of the distinct index-labels in `ixs` of the input not appearing in the output `iy`
- a tuple of tuples of locations of an index-label in the `ixs` in a list of all index-labels
- a tuple of locations of index-labels in `iy` in a list of all index-labels

where the list of all index-labels is simply the first  and the second output catenated and the second output catenated.
"
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

# dynamic EinIndexer
struct DynamicEinIndexer{N}
    locs::NTuple{N,Int}
    cumsize::NTuple{N, Int}
end

function dynamic_indexer(locs::NTuple{N,Int}, size::NTuple{N,Int}) where {N}
    N==0 && return DynamicEinIndexer{0}((),())
    DynamicEinIndexer{N}(locs, (1,TupleTools.cumprod(size[1:end-1])...))
end

getlocs(d::DynamicEinIndexer{N}) where {N} = d.locs

# get a subset of index
@inline subindex(indexer::DynamicEinIndexer, ind::CartesianIndex) = subindex(indexer, ind.I)
@inline subindex(indexer::DynamicEinIndexer{0}, ind::NTuple{N0,Int}) where N0 = 1
@inline @generated function subindex(indexer::DynamicEinIndexer{N}, ind::NTuple{N0,Int}) where {N,N0}
    ex = Expr(:call, :+, map(i->i==1 ? :(ind[indexer.locs[$i]]) : :((ind[indexer.locs[$i]]-1) * indexer.cumsize[$i]), 1:N)...)
    :(@inbounds $ex)
end
