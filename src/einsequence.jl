"""
    parse_nested(s::AbstractString (, iy = []))

return a contraction-tree consisting of `NestedEinsumConstructor`
objects as nodes and `IndexGroup` as leaves.
`s` should only contain the `ixs` part
"""

function parse_nested(s::AbstractString, iy = [])
    count(==('('),s) == count(==(')'),s) || throw(
        ArgumentError("Parentheses don't pair up in $s"))

    _, out = parse_parens(s, firstindex(s), 1)
    append!(out.iy, iy)
    filliys!(out)
    return construct(out)
end

"""
    parse_parens(s::AbstractString, i, narg)

parse one level of parens starting at index `i` where `narg` counts which tensor the
current group of indices, e.g. "ijk", belongs to.
Recursively calls itself for each new opening paren that's opened.
"""
function parse_parens(s::AbstractString, i, narg)
    out = NestedEinsumConstructor{Char}([], [], [])
    g = IndexGroup{Char}([],narg)

    while i <= lastindex(s)
        c = s[i]
        j = nextind(s,i)
        if c === '('
            # opening a parens means that next a new contraction is parsed
            j, out2, narg = parse_parens(s, j, narg)
            push!(out, out2)
            union!(out.inds, out2.inds)
        elseif c === ')' || c === ','
            # either closing a parens or a comma means that a tensors indices
            # are complete and can be added to the contraction-object
            if !isempty(g)
                push!(out, g)
                union!(out.inds, g.inds)
            end
            if  c === ','
                # comma implies that a new tensor is parsed for the current contraction
                narg += 1
                g = IndexGroup{Char}([], narg)
            else
                # parens implies that the current contraction is complete
                return j, out, narg
            end
        elseif isletter(c) # could allow isnumeric() too
            push!(g,c)
        else
            throw(ArgumentError("parsing $s failed, $c is not a valid entry"))
        end
        i = j
    end
    if !isempty(g)
        push!(out, g)
        union!(out.inds, g.inds)
    end
    return i, out
end

"""
    filliys!(neinsum::NestedEinsumConstructor)

goes through all `NestedEinsumConstructor` objects in the tree and saves the correct `iy` in them.
"""
function filliys!(neinsum)
    iy = neinsum.iy
    args = neinsum.args
    for i in 1:length(neinsum.args)
        arg = args[i]
        arg isa IndexGroup && continue
        union!(arg.iy, intersect(arg.inds, iy))
        for j in 1:length(neinsum.args)
            i === j && continue
            union!(arg.iy, intersect(arg.inds, args[j].inds))
        end
        filliys!(arg)
    end
    return neinsum
end

"""
    IndexGroup

Leaf in a contractiontree, contains the indices and the number of the tensor it
describes, e.g. in "ij,jk -> ik", indices "ik" belong to tensor `1`, so
would be described by IndexGroup(['i','k'], 1).
"""
struct IndexGroup{T}
    inds::Vector{T}
    n::Int
end

Base.push!(ig::IndexGroup, c) = (push!(ig.inds,c); ig)
Base.isempty(ig::IndexGroup) = isempty(ig.inds)

"""
    NestedEinsumConstructor

describes a (potentially) nested einsum. Important fields:
- `args`, vector of all inputs, either `IndexGroup` objects corresponding to tensors or `NestedEinsumConstructor`
- `iy`, indices of output
"""
struct NestedEinsumConstructor{T}
    args::Vector{Union{NestedEinsumConstructor{T}, IndexGroup{T}}}
    inds::Vector{T}
    iy::Vector{T}
end

Base.push!(neinsum::NestedEinsumConstructor, x) = (push!(neinsum.args,x); neinsum)

using MacroTools

function _nested_ein_macro(ex; einsum=:einsum)
    @capture(ex, (left_ := right_)) || throw(ArgumentError("expected A[] := B[]... "))
    @capture(left, Z_[leftind__] | [leftind__] ) || throw(
        ArgumentError("can't understand LHS, expected A[i,j] etc."))
    Z === nothing && @gensym Z
    primefix!(leftind)
    allinds = unique(leftind)

    MacroTools.postwalk(right) do x
        @capture(x, A_[inds__]) && union!(allinds, inds)
        x
    end
    primefix!(allinds)

    tensors = Symbol[]
    nein = parse_nested_expr(right, tensors, allinds)
    append!(nein.iy, indexin(leftind,allinds))
    filliys!(nein)
    snein = construct(nein)

    tensornames = map(esc, tensors)
    :($(esc(Z)) = $snein(($(tensornames...),)...))
end

function parse_nested_expr(expr, tensors, allinds)
    if @capture(expr, *(args__))
        einargs = map(x -> parse_nested_expr(x,tensors, allinds), args)
        intinds = union(mapreduce(x -> x.inds, vcat, einargs))
        return NestedEinsumConstructor{Int}(einargs, intinds, Int[])
    elseif @capture(expr, A_[inds__])
        push!(tensors,A)
        return IndexGroup{Int}(indexin(primefix!(inds), allinds), length(tensors))
    end
end

# the contraction tree
"""
    NestedEinsum{LT} <: AbstractEinsum

The abstract type for contraction trees. It has two subtypes, [`DynamicNestedEinsum`](@ref) and [`StaticNestedEinsum`](@ref).
"""
abstract type NestedEinsum{LT} <: AbstractEinsum end

"""
    DynamicNestedEinsum{LT} <: NestedEinsum{LT}
    DynamicNestedEinsum(args, eins)
    DynamicNestedEinsum{LT}(tensorindex::Int)

Einsum with contraction order, where the type parameter `LT` is the label type.
It has two constructors. One takes a `tensorindex` as input, which represents the leaf node in a contraction tree.
The other takes an iterable of type `DynamicNestedEinsum`, `args`, as the siblings, and `eins` to specify the contraction operation.
"""
struct DynamicNestedEinsum{LT} <: NestedEinsum{LT}
    args::Vector{DynamicNestedEinsum{LT}}
    tensorindex::Int  # -1 if not leaf
    eins::DynamicEinCode{LT}

    function DynamicNestedEinsum(args::Vector{DynamicNestedEinsum{LT}}, eins::DynamicEinCode{LT}) where LT
        @assert length(args) == length(getixsv(eins))
        new{LT}(args, -1, eins)
    end
    DynamicNestedEinsum{LT}(arg::Int) where LT = new{LT}(NestedEinsum{LT}[], arg)
end
function DynamicNestedEinsum(args, eins::DynamicEinCode{LT}) where LT
    DynamicNestedEinsum(collect(DynamicNestedEinsum{LT}, args), eins)
end
isleaf(ne::DynamicNestedEinsum) = ne.tensorindex != -1
siblings(ne::DynamicNestedEinsum) = ne.args
tensorindex(ne::DynamicNestedEinsum) = ne.tensorindex
rootcode(ne::DynamicNestedEinsum) = ne.eins

"""
    StaticNestedEinsum{LT,args,eins} <: NestedEinsum{LT}
    StaticNestedEinsum(args, eins)
    StaticNestedEinsum{LT}(tensorindex::Int)

Einsum with contraction order, where the type parameter `LT` is the label type,
`args` is a tuple of StaticNestedEinsum, `eins` is a `StaticEinCode` and leaf node is defined by setting `eins` to an integer.
It has two constructors. One takes a `tensorindex` as input, which represents the leaf node in a contraction tree.
The other takes an iterable of type `DynamicNestedEinsum`, `args`, as the siblings, and `eins` to specify the contraction operation.
"""
struct StaticNestedEinsum{LT,args,eins} <: NestedEinsum{LT}
    function StaticNestedEinsum(args::NTuple{N,StaticNestedEinsum{LT}}, eins::StaticEinCode{LT}) where {N,LT}
        @assert length(args) == length(getixs(eins))
        new{LT,args,eins}()
    end
    function StaticNestedEinsum{LT}(tensorindex::Int) where {LT}
        new{LT,(),tensorindex}()
    end
end
isleaf(::StaticNestedEinsum{LT,args,eins}) where {LT,args,eins} = eins isa Int
siblings(::StaticNestedEinsum{LT,args}) where {LT,args} = args
tensorindex(ne::StaticNestedEinsum{LT,args,eins}) where {LT,args,eins} = (@assert isleaf(ne); eins)
rootcode(::StaticNestedEinsum{LT,args,eins}) where {LT,args,eins} = eins

function Base.:(==)(a::NestedEinsum, b::NestedEinsum)
    siba, sibb = siblings(a), siblings(b)
    (isleaf(a) != isleaf(b) || length(siba) != length(sibb)) && return false
    ex = if isleaf(a)
        tensorindex(a) == tensorindex(b)
    else
        rootcode(a) == rootcode(b)
    end
    return ex && all(i->siba[i] == sibb[i], 1:length(siba))
end

# conversion
function StaticNestedEinsum(ne::DynamicNestedEinsum{LT}) where LT
    if isleaf(ne)
        StaticNestedEinsum{LT}(tensorindex(ne))
    else
        sib = siblings(ne)
        StaticNestedEinsum(ntuple(i->StaticNestedEinsum(sib[i]),length(sib)), StaticEinCode(rootcode(ne)))
    end
end
function DynamicNestedEinsum(ne::StaticNestedEinsum{LT}) where LT
    if isleaf(ne)
        DynamicNestedEinsum{LT}(tensorindex(ne))
    else
        DynamicNestedEinsum([DynamicNestedEinsum(s) for s in siblings(ne)], DynamicEinCode(rootcode(ne)))
    end
end
function NestedEinsum(args, eins::EinCode)
    eins isa DynamicEinCode ? DynamicNestedEinsum(args, eins) : StaticNestedEinsum(args, eins)
end

function construct(nein::NestedEinsumConstructor{T}) where T
    ixs = Tuple(map(extractixs, nein.args))
    iy = Tuple(nein.iy)
    eins = StaticEinCode{T,ixs,iy}()
    args = Tuple(map(x -> x isa NestedEinsumConstructor ? construct(x) : StaticNestedEinsum{T}(x.n), nein.args))
    return StaticNestedEinsum(args, eins)
end
extractixs(x::IndexGroup) = Tuple(x.inds)
extractixs(x::NestedEinsumConstructor) = Tuple(x.iy)

# For CuArrays, kwargs can be [`active_free`].
function (neinsum::NestedEinsum{LT})(@nospecialize(xs::AbstractArray...); size_info = nothing, kwargs...) where LT
    size_dict = size_info===nothing ? Dict{LT,Int}() : copy(size_info)
    get_size_dict!(neinsum, xs, size_dict)
    return einsum(neinsum, xs, size_dict; kwargs...)
end

function get_size_dict!(ne::NestedEinsum, @nospecialize(xs), size_info::Dict{LT}) where LT
    d = collect_ixs!(ne, Dict{Int,Vector{LT}}())
    ks = sort!(collect(keys(d)))
    ixs = [d[i] for i in ks]
    return get_size_dict_!(ixs, [collect(Int, size(xs[i])) for i in ks], size_info)
end

function einsum(neinsum::NestedEinsum, @nospecialize(xs::NTuple{N,AbstractArray} where N), size_dict::Dict)
    # do not use map because the static overhead is too large
    # do not use `setindex!` because we need to make the AD work
    mxs = Vector{AbstractArray}(undef, length(siblings(neinsum)))
    for (i, arg) in enumerate(siblings(neinsum))
        mxs = _safe_set(mxs, i, isleaf(arg) ? xs[tensorindex(arg)] : einsum(arg, xs, size_dict))
    end
    return einsum(rootcode(neinsum), (mxs...,), size_dict)
end

_safe_set(lst, i, y) = (lst[i] = y; lst)

# Better printing
struct LeafString
    str::String
end
function AbstractTrees.children(ne::NestedEinsum)
    [isleaf(item) ? LeafString(_join(getixs(rootcode(ne))[k])) : item for (k,item) in enumerate(siblings(ne))]
end

AbstractTrees.printnode(io::IO, e::LeafString) = print(io, e.str)
function AbstractTrees.printnode(io::IO, x::NestedEinsum)
    isleaf(x) ? print(io, tensorindex(x)) : print(io, rootcode(x))
end

function Base.show(io::IO, e::EinCode)
    s = join([_join(ix) for ix in getixs(e)], ", ") * " -> " * _join(getiy(e))
    print(io, s)
end
function Base.show(io::IO, e::NestedEinsum)
    print_tree(io, e)
end
Base.show(io::IO, ::MIME"text/plain", e::NestedEinsum) = show(io, e)
Base.show(io::IO, ::MIME"text/plain", e::EinCode) = show(io, e)
_join(ix) = isempty(ix) ? "" : join(ix, connector(eltype(ix)))
connector(::Type{Char}) = ""
connector(::Type{Int}) = "∘"
connector(::Type) = "-"

# flatten nested einsum
function _flatten(code::NestedEinsum, iy=nothing)
    isleaf(code) && return [tensorindex(code)=>iy]
    sibs = siblings(code)
    ixs = []
    for i=1:length(sibs)
        append!(ixs, _flatten(sibs[i], getixs(rootcode(code))[i]))
    end
    return ixs
end

flatten(code::EinCode) = code
function flatten(code::DynamicNestedEinsum{LT}) where LT
    ixd = Dict(_flatten(code))
    DynamicEinCode([ixd[i] for i=1:length(ixd)], collect(getiy(code.eins)))
end
function flatten(code::StaticNestedEinsum{LT}) where LT
    ixd = Dict(_flatten(code))
    StaticEinCode{LT,ntuple(i->(ixd[i]...,), length(ixd)), getiy(rootcode(code))}()
end

labeltype(::NestedEinsum{LT}) where LT = LT
function getixsv(ne::NestedEinsum{LT}) where LT
    if isleaf(ne)
        error("Can not call `getiyv` on leaf nodes!")
    end
    d = collect_ixs!(ne, Dict{Int,Vector{LT}}())
    ks = sort!(collect(keys(d)))
    return @inbounds [d[i] for i in ks]
end
function collect_ixs!(ne::NestedEinsum, d::Dict{Int,Vector{LT}}) where LT
    args = siblings(ne)
    @inbounds for i=1:length(args)
        arg = args[i]
        if isleaf(arg)
            d[tensorindex(arg)] = _collect(LT, getixs(rootcode(ne))[i])
        else
            collect_ixs!(arg, d)
        end
    end
    return d
end
function getiyv(ne::NestedEinsum{LT}) where LT
    if isleaf(ne)
        error("Can not call `getiyv` on leaf nodes!")
    end
    getiyv(rootcode(ne))
end
