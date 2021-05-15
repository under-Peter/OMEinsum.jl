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
    append!(out.iy, iy)
    filliys!(out)
    return out
end

"""
    parse_parens(s::AbstractString, i, narg)

parse one level of parens starting at index `i` where `narg` counts which tensor the
current group of indices, e.g. "ijk", belongs to.
Recursively calls itself for each new opening paren that's opened.
"""
function parse_parens(s::AbstractString, i, narg)
    out = NestedEinsum{Char}([], [], [])
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
struct IndexGroup{T}
    inds::Vector{T}
    n::Int
end

Base.push!(ig::IndexGroup, c) = (push!(ig.inds,c); ig)
Base.isempty(ig::IndexGroup) = isempty(ig.inds)

"""
    NestedEinsum

describes a (potentially) nested einsum. Important fields:
- `args`, vector of all inputs, either `IndexGroup` objects corresponding to tensors or `NestedEinsum`
- `iy`, indices of output
"""
struct NestedEinsum{T}
    args::Vector{Union{NestedEinsum{T}, IndexGroup{T}}}
    inds::Vector{T}
    iy::Vector{T}
end

Base.push!(neinsum::NestedEinsum, x) = (push!(neinsum.args,x); neinsum)

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
    snein = stabilize(nein)

    tensornames = map(esc, tensors)
    :($(esc(Z)) = $snein(($(tensornames...),)...))
end

function parse_nested_expr(expr, tensors, allinds)
    if @capture(expr, *(args__))
        einargs = map(x -> parse_nested_expr(x,tensors, allinds), args)
        intinds = union(mapreduce(x -> x.inds, vcat, einargs))
        return NestedEinsum{Int}(einargs, intinds, Int[])
    elseif @capture(expr, A_[inds__])
        push!(tensors,A)
        return IndexGroup{Int}(indexin(primefix!(inds), allinds), length(tensors))
    end
end


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

match_rule(code::Int64) = code
