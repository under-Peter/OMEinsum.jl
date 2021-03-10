# Unary operations are searched in the following order
# 0. special rules `Identity` and `Tr`,
# 1. rules reducing dimensions `PTrace` and `Sum`,
# 2. `Permutedims`,
# 3. `Repeat` and `Duplicate`,
#   - we use `loop_einsum` instead of using existing API,
#   - because it is has similar or even better performance.

# For unclassified unary rules
# 1. simplify the input and output patterns with `PTrace`,
# 2. the simplified pattern can be handled by `Sum` + `Permutedims`,
# 3. generate the correct output using `Repeat` and `Duplicate`,

struct Sum <: EinRule{1} end
struct Tr <: EinRule{1} end
struct Permutedims <: EinRule{1} end
struct PTrace <: EinRule{1} end
struct Identity <: EinRule{1} end
struct Repeat <: EinRule{1} end
struct Duplicate <: EinRule{1} end

"""
a einsum code is repeating some dimensions.
"""
function match_rule(::Type{Repeat}, ixs, iy)
    length(ixs) != 1 && return false
    ix, = ixs
    allunique(ix) && allunique(iy) && all(i -> i in iy, ix)
end

"""
a einsum code is trace
"""
function match_rule(::Type{Tr}, ixs, iy)
    iy == () &&
    length.(ixs) === (2,) &&
    ixs[1][1] == ixs[1][2]
end

"""
a einsum code is sum.
"""
function match_rule(::Type{Sum}, ixs, iy)
    length(ixs) != 1 && return false
    (ix,) = ixs
    allunique(ix) && allunique(iy) && all(i -> i in ix, iy)
end

"""
permutation rule
"""
function match_rule(::Type{Permutedims}, ixs, iy)
    length(ixs) == 1 || return false
    (ix,) = ixs
    length(ix) == length(iy) && allunique(ix) && allunique(iy) &&
     all(i -> i in iy, ix)
end

"""
Ptrace rule if all indices of one ix in ixs all appear in iy or
appear twice and don't appear in iy
"""
function match_rule(::Type{PTrace}, ixs, iy)
    length(ixs) == 1 || return false
    (ix,) = ixs
    all(iy) do i
        count(==(i), ix) == 1 && count(==(i), iy) == 1
    end || return false
    all(ix) do i
        ciy, cix = count.(==(i), (iy,ix))
        (ciy == 0 && cix == 2) ||
        (ciy == 1 && cix == 1) ||
        (ciy > 1)
    end || return false
end

function match_rule(::Type{Identity}, ixs, iy)
    ixs === (iy,) && allunique(iy)
end

const einsum_rules = [
    Identity,
    Permutedims,
    Tr,
    PTrace,
    Sum,
    ]

@doc raw"
    match_rule(ixs, iy)
    match_rule(code::EinCode{ixs, iy})
    match_rule(code::NestedEinCode)

go through all operations specified in the `einsum_rules`-vector and return
the first `T` for which `match_rule(T, ixs, iy)` returns true.
"
function match_rule(ixs, iy)
    # the first rule with the higher the priority
    for T in einsum_rules
        match_rule(T, ixs, iy) && return T()
    end
    return DefaultRule()
end

match_rule(code::EinCode{ixs, iy}) where {ixs, iy} = match_rule(ixs, iy)

# trace
function einsum(code::EinCode{(('i',),('i',)), ()}, xs::NTuple{2, Any}, size_dict)
    asarray(tr(xs[1]), xs[1])
end

using TensorOperations

function einsum(::PTrace, ::EinCode{ixs,iy}, xs::NTuple{<:Any, AbstractArray{<:BlasFloat}}, size_dict) where {ixs, iy}
    asarray(tensortrace(xs[1], ixs[1], iy), xs[1])
end

function einsum(sm::Sum, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    dims = (findall(i -> i ∉ iy, ixs[1])...,)
    (ix1,) = ixs
    ix1f = filter!(i -> i in iy, collect(ix1))
    # einsum(EinCode{((ix1f...,),),(iy)}(), res, size_dict)
    perm = map(i -> findfirst(==(i), ix1f), iy)
    res = dropdims(sum(xs[1], dims=dims), dims=dims)
    if perm == iy
        @debug "Sum" ixs => iy size.(xs)
        res
    else
        @debug "Sum permutedims" ixs => iy size.(xs) perm
        tensorpermute(res, perm)
    end
end

function einsum(::Permutedims, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    (ix,) = ixs
    (x,) = xs
    perm = map(i -> findfirst(==(i), ix), iy)
    @debug "Permutedims" ix => iy size(xs[1]) perm
    return tensorpermute(x, perm)
end

function einsum(::Identity, ::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    xs[1]
end

# the fallback
function einsum(::DefaultRule, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    @debug "DefaultRule loop_einsum" ixs => iy size.(xs)
    loop_einsum(code, xs, size_dict)
end

function einsum(::PTrace, code::EinCode{ixs, iy}, xs::NTuple{NT, Any}, size_dict) where {ixs, iy, NT}
    @debug "PTrace loop_einsum" ixs => iy size.(xs)
    loop_einsum(code, xs, size_dict)
end

@generated function einsum(::BatchedContract, ::EinCode{ixs,iy}, xs::NTuple{<:Any, AbstractArray{<:BlasFloat}}, size_dict) where {ixs, iy}
    quote
        ixs1, xs1 = _preprocess_dupindices($(Val(ixs[1])), xs[1])
        ixs2, xs2 = _preprocess_dupindices($(Val(ixs[2])), xs[2])
        @debug "BatchedContract" ixs => iy ixs1 ixs2 size(xs1) size(xs2)
        batched_contract(ixs1, xs1, ixs2, xs2, $(Val(iy)))
    end
end