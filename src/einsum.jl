using TupleTools, Base.Cartesian

function outindsfrominput(ixs)
    allixs = vcat(collect.(ixs)...)
    iy = sort!(filter!(x -> count(==(x), TupleTools.vcat(ixs...)) == 1, allixs))
    return tuple(iy...)
end

function einsum(s::AbstractString, xs)
    s = replace(s, " " => "")
    m = match(r"([a-z,]+)->([a-z]*)", s)
    m == nothing && throw(ArgumentError("invalid einsum specification $s"))
    sixs, siy = m.captures
    iy  = Tuple(siy)
    ixs = Tuple(Tuple(ix) for ix in split(sixs,','))
    return einsum(ixs, xs, iy)
end

einsum(ixs, xs) = einsum(ixs, xs, outindsfrominput(ixs))

@doc raw"
    einsum(ixs, xs, out)
return the tensor that results from contracting the tensors `xs` according
to their indices `ixs`, where all indices that do not appear in the output are
summed over. The indices are contracted in the order implied by their numerical value,
smaller first.
The result is permuted according to `out`.

- `ixs` - tuple of tuple of integers that label all indices of a tensor.
       Indices that appear twice (in different tensors) are summed over

- `xs` - tuple of tensors

- `out` - tuple of integers that should correspond to remaining indices in `ixs` after contractions.


# example
```jldoctest; setup = :(using OMEinsum)
julia> a = rand(2,2);

julia> b = rand(2,2);

julia> einsum(((1,2),(2,3)), (a, b), (1,3)) ≈ a * b
true

julia> einsum(((1,2),(2,3)), (a, b), (3,1)) ≈ permutedims(a * b, (2,1))
true
```
"
function einsum(ixs, xs, iy)
    checkargs(ixs, xs, iy)
    ops = opsfrominds(ixs, iy)
    evaluateall(ixs, xs, ops, iy)
end

einsumopt(ixs, xs) = einsumopt(ixs, xs, outindsfrominput(ixs))
@doc raw"
    meinsumopt(ixs, xs, iy)
returns the result of the einsum operation implied by `ixs`, `iy` but
evaluated in the optimal order according to `meinsumcost`.
"
function einsumopt(ixs, xs, iy)
    checkargs(ixs, xs, iy)
    ops = optimalorder(ixs, xs, iy)
    evaluateall(ixs, xs, ops, iy)
end

function checkargs(ixs, xs, iy)
    length(ixs) == length(xs) || throw(
        ArgumentError("Number of indices and tensors not the same"))
    foreach(ixs, xs) do ix, x
        length(ix) == ndims(x) || throw(
        ArgumentError("Indices $ix are invalid for a tensor with ndims = $(ndims(x))"))
    end
end

function einsumexp(ixs::NTuple{N, NTuple{M, T} where M},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                iy::NTuple{<:Any,T}) where {N,T}
    out = outputtensor(xs, ixs, iy)
    einsumexp!(ixs, xs, iy, out)
end

function outputtensor(xs, ixs, iy)
    T = mapreduce(eltype, promote_type, xs)
    sizes = TupleTools.vcat(size.(xs)...)
    indices = TupleTools.vcat(ixs...)
    outdims = map(x -> sizes[findfirst(==(x), indices)], iy)
    zeros(T, outdims...)
end
