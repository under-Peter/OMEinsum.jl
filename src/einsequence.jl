"""
    parse_nested(s::AbstractString (, iy = []))
return a contraction-tree consisting of `NestedEinsum`
objects as nodes and `IndexGroup` as leaves.
`s` should only contain the `ixs` part
"""

function parse_nested(s::AbstractString, iy = [])
    count(==('('),s) == count(==(')'),s) || throw(
        ArgumentError("Parentheses don't pair up in $s"))

    _, out = parse_parens(s, firstindex(s), 1)
    out.iy = iy
    filliys!(out)
    return stabilize(out)
end

"""
    parse_parens(s::AbstractString, i, narg)
parse one level of parens starting at index `i` where `narg` counts which tensor the
current group of indices, e.g. "ijk", belongs to.
Recursively calls itself for each new opening paren that's opened.
"""
function parse_parens(s::AbstractString, i, narg)
    out = NestedEinsum([], [], [])
    g = IndexGroup([],narg)

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
                g = IndexGroup([], narg)
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
    filliys!(neinsum::NestedEinsum)
goes through all `NestedEinsum` objects in the tree and saves the correct `iy` in them.
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
struct IndexGroup
    inds::Vector{Char}
    n::Int
end

Base.push!(ig::IndexGroup, c::Char) = (push!(ig.inds,c); ig)
Base.isempty(ig::IndexGroup) = isempty(ig.inds)

"""
    NestedEinsum
describes a (potentially) nested einsum. Important fields:
- `args`, vector of all inputs, either `IndexGroup` objects corresponding to tensors or `NestedEinsum`
- `iy`, indices of output
"""
mutable struct NestedEinsum
    args::Vector{Union{NestedEinsum, IndexGroup}}
    inds::Vector{Char}
    iy::Vector{Char}
end

Base.push!(neinsum::NestedEinsum, x) = (push!(neinsum.args,x); neinsum)

"""
apply a NestedEinsum to arguments evaluates the nested einsum
"""
function (neinsum::NestedEinsum)(xs...; size_dict = nothing)
    ixs = Tuple(map(extractixs, neinsum.args))
    iy = Tuple(neinsum.iy)
    EinCode(ixs,iy)(map(arg -> extractxs(xs, arg), neinsum.args)...)
end

"""
extract the indices of the tensor that is associated with x (if x isa IndexGroup)
or results from x (if x isa NestedEinsum)
"""
extractixs(x::IndexGroup) = Tuple(x.inds)
extractixs(x::NestedEinsum) = Tuple(x.iy)

"""
extract the tensor associated with x (if x isa IndexGroup)
or evaluate and return the tensor associated with x (if x isa NestedEinsum)
"""
extractxs(xs, x::NestedEinsum) = x(xs...)
extractxs(xs, x::IndexGroup) = xs[x.n]

mutable struct NestedEinsumStable{T,S,N}
    args::S
    eins::T
end

function stabilize(nein::OMEinsum.NestedEinsum)
    ixs = Tuple(map(OMEinsum.extractixs, nein.args))
    iy = Tuple(nein.iy)
    eins = EinCode{ixs,iy}()
    args = Tuple(map(x -> x isa OMEinsum.NestedEinsum ? stabilize(x) : x.n,nein.args))
    return NestedEinsumStable{typeof(eins), typeof(args), length(iy)}(args, eins)
end

function (neinsum::NestedEinsumStable{<:Any,<:Any,N})(xs...; size_dict = nothing) where N
    mxs = map(x -> extractxs(xs, x), neinsum.args)
    neinsum.eins(mxs...)
end

extractxs(xs, x::NestedEinsumStable) = x(xs...)
extractxs(xs, i::Int) = xs[i]
