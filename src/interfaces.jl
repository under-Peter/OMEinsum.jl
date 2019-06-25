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
    N != X && throw(ArgumentError(""))

    # check tensor orders
    foreach(ixs, xs) do ix, x
        length(ix) == ndims(x) || throw(ArgumentError(""))
    end
    sd = get_size_dict!(Dict{T,Int}(), ixs, xs)
    allixs = TupleTools.flatten(ixs)
    allsxs = TupleTools.flatten(size.(xs))
    foreach(allixs, allsxs) do i, s
        sd[i] == s || throw(DimensionMismatch(""))
    end
    return sd
end

function get_size_dict!(size_dict::Dict{T,Int}, ixs::NTuple{N, NTuple{M, T} where M}, xs::NTuple{X, AbstractArray}) where {T,X,N}
    @inbounds for i = 1:N
        for (n, leg) in zip(size(xs[i]), ixs[i])
            size_dict[leg] = n
        end
    end
    return size_dict
end
