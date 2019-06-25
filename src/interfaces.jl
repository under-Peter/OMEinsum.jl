export @ein_str

# TODO: delete this kind of interface completely?
function einsumexp!(ixs::NTuple{N, NTuple{M, IT} where M},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                iy::NTuple{L,IT},
                y::AbstractArray{T,L}) where {N,L,T,IT <: Union{AbstractChar,Integer}}
    size_dict = get_size_dict((ixs..., iy), (xs..., y))
    einsumexp!(EinCode(ixs, iy), xs, y, size_dict)
end

function einsumexp!(code::EinCode{ixs, iy},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                y::AbstractArray{T,L}) where {N,L,T, ixs, iy}
    size_dict = get_size_dict((ixs..., iy), (xs..., y))
    einsumexp!(code, xs, y, size_dict)
end

macro ein_str(s::AbstractString)
    s = replace(s, " " => "")
    m = match(r"([a-z,]+)->([a-z]*)", s)
    m == nothing && throw(ArgumentError("invalid einsum specification $s"))
    sixs, siy = m.captures
    iy  = Tuple(siy)
    ixs = Tuple(Tuple(ix) for ix in split(sixs,','))
    return EinCode(ixs, iy)
end

(code::EinCode)(xs...; size_dict=nothing) where {T, N} = einsum(code, xs, size_dict)

einsum(code::EinCode{ixs, iy}, xs) where {ixs, iy} = einsum(code, xs, nothing)

function einsum(code::EinCode{ixs, iy}, xs, ::Nothing) where {ixs, iy}
    einsum(code, xs, get_size_dict(ixs, xs))
end

function infer_y_size(xs, ixs, iy)
    sizes = TupleTools.vcat(size.(xs)...)
    indices = TupleTools.vcat(ixs...)
    map(iy) do x
        i = findfirst(==(x), indices)
        i==nothing && throw(ArgumentError("Size of output index $x can not be infered."))
        return sizes[i]
    end
end

"""get the dictionary of `index=>size`, error if there are conflicts"""
function get_size_dict(ixs::NTuple{N, NTuple{M, T} where M}, xs::NTuple{X, AbstractArray}) where {N,T,X}
    # check size of input tuples
    N != X && throw(ArgumentError("$X tensors labelled by $N indices"))

    # check tensor orders
    foreach(ixs, xs) do ix, x
        length(ix) == ndims(x) || throw(
            ArgumentError("indices $ix invalid for tensor with ndims = $(ndims(x))"))
    end
    sd = IndexSize(ixs, xs)
    dimensionsmatch(sd)
    return sd
end

struct IndexSize{N,T}
    k::NTuple{N,T}
    v::NTuple{N,Int32}
end

function IndexSize(ixs, xs)
    k = TupleTools.flatten(ixs)
    v = TupleTools.flatten(map(size,xs))
    T, N = eltype(k), length(k)
    IndexSize{N,T}(k,v)
end

Base.getindex(inds::IndexSize{N,T},i::T) where {N,T} = inds.v[findfirst(==(i), inds.k)]

function dimensionsmatch(inds::IndexSize)
    for (c,i) in enumerate(inds.k)
        j = findnext(==(i), inds.k, c+1)
        j != nothing && inds.v[c] != inds.v[j] && return throw(
            DimensionMismatch("index $i has incosistent sizes"))
    end
    return true
end
