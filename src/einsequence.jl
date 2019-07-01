# macro eins_str(s::AbstractString)
#     s = replace(s, " " => "")
#     m = match(r"([\(\)a-z,]+)->([a-z]*)", s)
#     m == nothing && throw(ArgumentError("invalid einsum specification $s"))
#     sixs, siy = m.captures
#     iy  = Tuple(siy)
#     ixs = Tuple(Tuple(ix) for ix in split(sixs,','))
#     return EinCode(ixs, iy)
# end


function parse_nested(s::AbstractString, iy = [])
    _, out = parse_level(s, firstindex(s), 1)
    out.iy = iy
    filliys!(out)
    return out
end

function parse_level(s::AbstractString, i, narg)
    out = Contraction([], 0, Set(), [])
    g = IndexGroup([],narg)
    while i <= lastindex(s)
        c = s[i]
        j = nextind(s,i)
        if c === '('
            j, out2, narg = parse_level(s, j, narg)
            out = push!(out, out2)
            union!(out.indswithin, out2.indswithin)
            out.nargs += out2.nargs
        elseif c === ')' || c === ','
            if !isempty(g)
                push!(out, g)
                out.nargs += 1
                union!(out.indswithin, g.inds)
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
            error()
        end
        i = j
    end
    if !isempty(g)
        out.nargs += 1
        push!(out, g)
        union!(out.indswithin, g.inds)
    end
    return i, out
end

function filliys!(cont)
    iy = cont.iy
    args = cont.args
    for i in 1:length(cont.args)
        arg = args[i]
        arg isa IndexGroup && continue
        union!(arg.iy, intersect(arg.indswithin, iy))
        for j in 1:length(cont.args)
            i === j && continue
            if args[j] isa IndexGroup
                union!(arg.iy, intersect(arg.indswithin, args[j].inds))
            else
                union!(arg.iy, intersect(arg.indswithin, args[j].indswithin))
            end
        end
        filliys!(arg)
    end
    return cont
end

struct IndexGroup
    inds::Vector{Char}
    n::Int
end

Base.push!(ig::IndexGroup, c::Char) = (push!(ig.inds,c); ig)
Base.isempty(ig::IndexGroup) = isempty(ig.inds)

mutable struct Contraction
    args::Vector{Union{Contraction, IndexGroup}}
    nargs::Int
    indswithin::Set{Char}
    iy::Vector{Char}
end

Base.push!(cont::Contraction, x) = (push!(cont.args,x); cont)

@time parse_nested("((ij,iab),jcd),afce", collect("bdfe"))
@time parse_nested("(ij,jk),kl", collect("im"))
@time parse_nested("(ij,jk),(kl,lm)", collect("im"))

using BenchmarkTools
@btime parse_nested("((ij,iab),jcd),afce", collect("bdfe"))
@btime parse_nested("(ij,jk),kl", collect("im"))
@btime parse_nested("(ij,jk),(kl,lm)", collect("im"))


using OMEinsum
function foo(cont::Contraction)
    ixs = Tuple(map(extractixs,cont.args))
    iy = Tuple(cont.iy)
    Expr(:call, :EinCode, ixs, iy)
end

extractixs(x::IndexGroup) = Tuple(x.inds)
extractixs(x::Contraction) = Tuple(x.iy)

cont =  parse_nested("(ij,jk),(kl,lm)", collect("im"))
foo(cont) |> eval
eval(foo(cont))(rand(2,2),rand(2,2))

function bar(x)
    :(y -> y * $x)
end

function barbar(::Val{x}) where x
    bar(x)
end

eval(bar(2))(2)
barbar(Val(2))

Foo(Foo(1,2),3)
x -> foo(foo(x[1],x[2]), x[3])
