struct SlicedEinsum{LT, Ein} <: AbstractEinsum
    slicing::Vector{LT}
    eins::Ein
end
Base.:(==)(se::SlicedEinsum, se2::SlicedEinsum) = se.slicing == se2.slicing && se.eins == se2.eins

# Iterate over tensor network slices, its iterator interface returns `slicemap` as a Dict
# slice and fill tensors with
# * take_slice(x, label_of_x, slicemap)
# * fill_slice!(x, label_of_x, x_slice, slicemap)
struct SliceIterator{LT,IT<:CartesianIndices}
    ixsv::Vector{Vector{LT}}
    iyv::Vector{LT}
    sliced_labels::Vector{LT}
    indices::IT
    size_dict_sliced::Dict{LT,Int}
end

function SliceIterator(se::SlicedEinsum, size_dict::Dict{LT}) where LT
    iyv = getiyv(se)
    ixsv = getixsv(se)
    return SliceIterator(ixsv, iyv, se.slicing, size_dict)
end

function SliceIterator(ixsv, iyv, legs, size_dict::Dict{LT}) where LT
    n = length(legs)
    size_dict_sliced = copy(size_dict)
    sliced_sizes = Vector{Int}(undef, n)
    sliced_labels = Vector{LT}(undef, n)
    for i = 1:n
        l = legs[i]
        sliced_sizes[i] = size_dict[l]
        sliced_labels[i] = l
        size_dict_sliced[l] = 1
    end
    indices = CartesianIndices((sliced_sizes...,))
    SliceIterator(ixsv, iyv, sliced_labels, indices, size_dict_sliced)
end
Base.length(si::SliceIterator) = length(si.indices)
Base.eltype(::Type{SliceIterator{LT,IT}}) where {LT,IT} = Dict{LT,Int}

# returns `slicemap` as a Dict
function Base.iterate(si::SliceIterator)
    ci, cistate = iterate(si.indices)
    slicemap = Dict(zip(si.sliced_labels, ones(Int,length(si.sliced_labels))))
    slicemap, (1,(ci,cistate),slicemap)
end
function Base.iterate(si::SliceIterator, state)
    i, (ci,cistate), slicemap = state
    if i >= length(si.indices)
        return nothing  # NOTE: ci is same as cistate
    else
        ci, cistate = iterate(si.indices, cistate)
        for (l, v) in zip(si.sliced_labels, ci.I)
            slicemap[l] = v
        end
        return slicemap, (i+1, (ci,cistate), slicemap)
    end
end
function Base.getindex(si::SliceIterator, indices...)
    ci = si.indices[indices...]
    slicemap = Dict(zip(si.sliced_labels, ci.I))
    return slicemap
end

function take_slice(x, ix, slicemap::Dict)
    slices = map(l->haskey(slicemap, l) ? slicemap[l] : Colon(), ix)
    if all(x->x isa Integer, slices)
        return copy(view(x,slices...))
    else
        return x[slices...]
    end
end
function fill_slice!(x, ix, chunk, slicemap::Dict)
    if ndims(x) == 0
        x .+= chunk  # to avoid CUDA `getindex!`.
    else
        slices = map(l->haskey(slicemap, l) ? slicemap[l] : Colon(), ix)
        view(x, slices...) .+= chunk
    end
    return x
end
function view_slice(x, ix, slicemap::Dict)
    if ndims(x) == 0
        return x
    else
        slices = map(l->haskey(slicemap, l) ? slicemap[l] : Colon(), ix)
        return view(x, slices...)
    end
end

function (se::SlicedEinsum{LT,ET})(@nospecialize(xs::AbstractArray...); size_info = nothing, kwargs...) where {LT, ET}
    # get size
    size_dict = size_info===nothing ? Dict{labeltype(se),Int}() : copy(size_info)
    get_size_dict!(se, xs, size_dict)
    # compute
    return einsum(se, xs, size_dict; kwargs...)
end

function einsum!(se::SlicedEinsum, @nospecialize(xs::NTuple{N,AbstractArray} where N), y, sx, sy, size_dict::Dict)
    length(se.slicing) == 0 && return einsum!(se.eins, xs, y, sx, sy, size_dict)
    iszero(sy) ? fill!(y, zero(eltype(y))) : rmul!(y, sy)
    it = SliceIterator(se, size_dict)
    eins_sliced = drop_slicedim(se.eins, se.slicing)
    for slicemap in it
        xsi = ntuple(i->take_slice(xs[i], it.ixsv[i], slicemap), length(xs))
        einsum!(eins_sliced, xsi, view_slice(y, it.iyv, slicemap), sx, true, it.size_dict_sliced)
    end
    return y
end
function einsum(se::SlicedEinsum, @nospecialize(xs::NTuple{N,AbstractArray} where N), size_dict::Dict)
    length(se.slicing) == 0 && return einsum(se.eins, xs, size_dict)
    it = SliceIterator(se, size_dict)
    res = get_output_array(xs, getindex.(Ref(size_dict), it.iyv))
    eins_sliced = drop_slicedim(se.eins, se.slicing)
    for slicemap in it
        # NOTE: @debug will break Zygote
        # @debug "computing slice $k/$(length(it))"
        xsi = ntuple(i->take_slice(xs[i], it.ixsv[i], slicemap), length(xs))
        resi = einsum(eins_sliced, xsi, it.size_dict_sliced)
        res = fill_slice!(res, it.iyv, resi, slicemap)
    end
    return res
end

function drop_slicedim(ne::NestedEinsum, slices::Vector)
    isleaf(ne) && return ne
    eins = rootcode(ne)
    ixs = map(ix->filter(∉(slices), ix), getixsv(eins))
    iy = filter(∉(slices), getiyv(eins))
    NestedEinsum(map(arg->drop_slicedim(arg, slices), siblings(ne)), similar_eincode(eins, ixs, iy))
end
similar_eincode(::DynamicEinCode, ixs, iy) = DynamicEinCode(ixs, iy)
similar_eincode(::StaticEinCode{LT}, ixs, iy) where LT = StaticEinCode{LT,(Tuple.(ixs)...,), (iy...,)}()

flatten(se::SlicedEinsum) = flatten(se.eins)
labeltype(::SlicedEinsum{LT}) where LT = LT
get_size_dict!(se::SlicedEinsum, xs, size_info::Dict) = get_size_dict!(se.eins, xs, size_info)
getixsv(se::SlicedEinsum) = getixsv(se.eins)
getiyv(se::SlicedEinsum) = getiyv(se.eins)
