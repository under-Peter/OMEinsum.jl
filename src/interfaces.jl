function parseeinsumsstring(s::AbstractString)
    s = replace(s, " " => "")
    m = match(r"([a-z,]+)->([a-z]*)", s)
    m == nothing && throw(ArgumentError("invalid einsum specification $s"))
    sixs, siy = m.captures
    iy  = Tuple(siy)
    ixs = Tuple(Tuple(ix) for ix in split(sixs,','))
    return (ixs, iy)
end

function einsum(s::AbstractString, xs)
    ixs, iy = parseeinsumsstring(s)
    return einsum(ixs, xs, iy)
end

function einsumexp!(ixs::NTuple{N, NTuple{M, IT} where M},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                iy::NTuple{L,IT},
                y::AbstractArray{T,L}) where {N,L,T,IT <: Union{AbstractChar,Integer}}
    size_dict = get_size_dict((ixs..., iy), (xs..., y))
    einsumexp!(EinCode(ixs, iy), xs, y, size_dict)
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
get_size_dict(ixs::NTuple{N, NTuple{M, T} where M} where N, xs) where T = get_size_dict!(Dict{T,Int}(), ixs, xs)
function get_size_dict!(size_dict::Dict{T,Int}, ixs::NTuple{N, NTuple{M, T} where M} where N, xs) where T
    nt = length(ixs)
    @inbounds for i = 1:nt
        for (N, leg) in zip(size(xs[i]), ixs[i])
            if haskey(size_dict, leg)
                size_dict[leg] == N || throw(DimensionMismatch("size of index($leg) does not match."))
            else
                size_dict[leg] = N
            end
        end
    end
    return size_dict
end
