"""
    parse_nested(s::AbstractString (, iy = []))
return a contraction-tree consisting of `NestedEinsum`
objects as nodes and `IndexGroup` as leaves.
`s` should only contain the `ixs` part
"""

function parse_nested(s::AbstractString, iy = [])
    count(==('('),s) == count(==(')'),s) || throw(
        ArgumentError("Parentheses don't pair up in $s"))
    _, out = parse_level(s, firstindex(s), 1)
    out.iy = iy
    filliys!(out)
    return out
end

"""
    parse_level(s::AbstractString, i, narg)
parse one level of parantheses starting at index `i` where `narg` counts which tensor the
current group of indices, e.g. "ijk", belongs to.
Recursively calls itself for each new parantheses that's opened.
"""
function parse_level(s::AbstractString, i, narg)
    out = NestedEinsum([], 0, Vector(), [])
    g = IndexGroup([],narg)
    while i <= lastindex(s)
        c = s[i]
        j = nextind(s,i)
        if c === '('
            j, out2, narg = parse_level(s, j, narg)
            out = push!(out, out2)
            union!(out.inds, out2.inds)
            out.nargs += out2.nargs
        elseif c === ')' || c === ','
            if !isempty(g)
                push!(out, g)
                out.nargs += 1
                union!(out.inds, g.inds)
            end
            if  c === ','
                narg += 1
                g = IndexGroup([], narg)
            else
                return j, out, narg
            end
        elseif isletter(c)
            g = push!(g,c)
        else
            throw(ArgumentError("parsing $s failed, $c is not a valid entry"))
        end
        i = j
    end
    if !isempty(g)
        out.nargs += 1
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
    nargs::Int
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

# @time parse_nested("((ij,iab),jcd),afce", collect("bdfe"))
# @time parse_nested("(ij,jk),kl", collect("il"))
# @time parse_nested("((ij,jk),kl)", collect("il"))
# test =  parse_nested("(ij,jk),kl", collect("il"))
# @time parse_nested("(ij,jk),(kl,lm)", collect("im"))
#
# using BenchmarkTools
# @btime parse_nested("((ij,iab),jcd),afce", collect("bdfe"))
# @btime parse_nested("(ij,jk),kl", collect("im"))
# @btime parse_nested("(ij,jk),(kl,lm)", collect("im"))
#
# nein1 = parse_nested("(ij,jk),kl", collect("il"))
# nein2 = parse_nested("((ij,jk),kl)", collect("il"))
# a, b, c = rand(2,2), rand(2,2), rand(2,2)
#
# nein1(a,b,c) ≈ nein2(a,b,c)
#
#
#
#
# parse_nested("(ij,jk),kl", collect("il"))(a,b,c) ≈ ein"ij,jk,kl -> il"(a,b,c)
# χ = 100
# a, b, c = [rand(χ,χ) for _ in 1:3]
# parse_nested("(ij,jk),kl", collect("il"))(a,b,c) ≈ ein"ij,jk,kl -> il"(a,b,c)
# @time parse_nested("(ij,jk),kl", collect("il"))(a,b,c)
# @time ein"ij,jk,kl -> il"(a,b,c)
#
# using BenchmarkTools
# χ = 20
# a, b, c = [rand(χ,χ) for _ in 1:3]
# @btime parse_nested("(ij,jk),kl", collect("il"))($a,$b,$c)
# @btime ein"ij,jk,kl -> il"($a,$b,$c)
# @btime ein"ij,jk -> ik"($a,$b)
# @btime $a * $b * $c
