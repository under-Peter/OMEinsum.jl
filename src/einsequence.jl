export NestedEinsum

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

struct NestedEinsum
    args
    eins
end

function construct(nein::NestedEinsumConstructor{T}) where T
    ixs = Tuple(map(extractixs, nein.args))
    iy = Tuple(nein.iy)
    eins = StaticEinCode{ixs,iy}()
    args = Tuple(map(x -> x isa NestedEinsumConstructor ? construct(x) : x.n,nein.args))
    return NestedEinsum(args, eins)
end
extractixs(x::IndexGroup) = Tuple(x.inds)
extractixs(x::NestedEinsumConstructor) = Tuple(x.iy)

function (neinsum::NestedEinsum)(xs...; size_info = nothing)
    mxs = map(x -> extractxs(xs, x), neinsum.args)
    neinsum.eins(mxs...; size_info=size_info)
end

extractxs(xs, x::NestedEinsum) = x(xs...)
extractxs(xs, i::Int) = xs[i]


function match_rule(code::NestedEinsum)
    return (match_rule(code.eins), match_rule.(code.args))
end

match_rule(code::Int64) = code


# Better printing
using AbstractTrees

function AbstractTrees.children(ne::NestedEinsum)
    d = Dict()
    for (k,item) in enumerate(ne.args)
        d[k] = item isa Integer ? _join(OMEinsum.getixs(ne.eins)[k]) : item
    end
    d
end

function AbstractTrees.printnode(io::IO, x::String)
    print(io, x)
end
function AbstractTrees.printnode(io::IO, x::NestedEinsum)
    print(io, x.eins)
end

function Base.show(io::IO, @nospecialize(e::EinCode))
    s = join([_join(ix) for ix in getixs(e)], ", ") * " -> " * _join(getiy(e))
    print(io, s)
end
function Base.show(io::IO, e::NestedEinsum)
    print_tree(io, e)
end
Base.show(io::IO, ::MIME"text/plain", e::NestedEinsum) = show(io, e)
Base.show(io::IO, ::MIME"text/plain", @nospecialize(e::EinCode)) = show(io, e)
_join(ix::NTuple{0}) = ""
_join(ix::NTuple{N,Char}) where N = join(ix, "")
_join(ix::NTuple{N,Int}) where N = join(ix, "∘")
_join(ix::NTuple{N,Any}) where N = join(ix, "-")


# flattten nested einsum
function _flatten(code::NestedEinsum, iy=nothing)
    ixs = []
    for i=1:length(code.args)
        append!(ixs, _flatten(code.args[i], OMEinsum.getixs(code.eins)[i]))
    end
    return ixs
end
_flatten(i::Int, iy) = [i=>iy]

flatten(@nospecialize(code::EinCode)) = code
function flatten(code::NestedEinsum)
    ixd = Dict(_flatten(code))
    EinCode(([ixd[i] for i=1:length(ixd)]...,), OMEinsum.getiy(code.eins))
end