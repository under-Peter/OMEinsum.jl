# for unary operations
# overhead ~ 2.3us
# @benchmark OMEinsum.einsum(DefaultRule(), $((('a', 'a', 'b'),)), $(('c', 'b','a')), (x,), $(Dict('a'=>1, 'b'=>1, 'c'=>1))) setup=(x=randn(1,1,1))
function einsum!(ixs, iy, @nospecialize(xs::NTuple{1, Any}), @nospecialize(y), sx, sy, size_dict::Dict{LT}) where LT
    ix, x = ixs[1], xs[1]
    @debug "compiling unary" ix => iy size(x)
    ix_unique = _unique(LT, ix)
    iy_unique = _unique(LT, iy)
    iy_a = filter(i->i ∈ ix, iy_unique)
    do_diag = length(ix_unique) != length(ix)
    do_duplicate = length(iy_unique) != length(iy)
    do_repeat = length(iy_a) != length(iy_unique)

    # diag
    if do_diag
        x_unique = similar(x, [size_dict[l] for l in ix_unique]...)
        unary_einsum!(Diag(), ix, (ix_unique...,), x, x_unique, true, false)
    else
        x_unique = x
    end

    # sum/permute
    if length(ix_unique) != length(iy_a)
        y_a = similar(x, [size_dict[l] for l in iy_a]...)
        unary_einsum!(Sum(), (ix_unique...,), (iy_a...,), x_, iy_a, true, false)
    elseif ix_unique != iy_a
        y_a = similar(x, [size_dict[l] for l in iy_a]...)
        unary_einsum!(Permutedims(), (ix_unique...,), (iy_a...,), x_, iy_a, true, false)
    else
        y_a = x_
    end
    # repeat indices
    # TODO: fix, should copy to y
    if do_repeat
        y_unique = similar(y, [size_dict[l] for l in iy_unique]...)
        unary_einsum!(Repeat(), (iy_a...,), (iy_unique...,), y_a, y_unique, true, false)
    else
        y_unique = y_a
    end
    # duplicate dimensions
    if do_duplicate
        return unary_einsum!(Duplicate(), (iy_unique...,), iy, y_unique, y, sx, sy)
    else
        return @addmul! sy * y + sx * y_unique
    end
end

# there are too many combination in the binary case, so nospecialize
function einsum!(ixs, iy, @nospecialize(xs::NTuple{2, Any}), @nospecialize(y), sx, sy, size_dict::Dict{LT}) where LT
    @debug "compiling binary" ixs => iy size.(xs)
    ix1, ix2 = ixs
    x1, x2 = xs
    c1, c2, cy, s1, s2, i1, i2, iyb = analyze_binary(_collect(LT,ix1), _collect(LT,ix2), _collect(LT,iy), size_dict)
    rule = SimpleBinaryRule{(i1...,), (i2...,), (iyb...,)}()
    x1 = simplify_unary(_collect(LT,ix1), c1, x1, size_dict)
    x2 = simplify_unary(_collect(LT,ix2), c2, x2, size_dict)
    x1_ = reshape(x1, s1...)
    x2_ = reshape(x2, s2...)
    @debug rule size.((x1_, x2_))
    y_ = reshape(einsum(rule, (x1_, x2_)), [size_dict[x] for x in cy]...)
    y .= expand_unary(cy, _collect(LT,iy), y_, size_dict)
    return y
end

@doc raw"
    match_rule(ixs, iy)
    match_rule(code::EinCode)

Returns the rule that matches, otherwise use `DefaultRule` - the slow `loop_einsum` backend.
"
function match_rule(ixs, iy)
    if length(ixs) == 1
        return match_rule_unary(ixs[1], iy)
    elseif length(ixs) == 2
        return match_rule_binary(ixs[1], ixs[2], iy)
    else
        return DefaultRule()
    end
end

function match_rule_unary(ix, iy)
    Nx = length(ix)
    Ny = length(iy)
    # the first rule with the higher the priority
    if Ny == 0 && Nx == 2 && ix[1] == ix[2]
        return Tr()
    elseif allunique(iy)
        if ix == iy
            return Identity()
        elseif allunique(ix)
            if Nx == Ny
                if all(i -> i in iy, ix)
                    return Permutedims()
                else  # e.g. (abcd->bcde)
                    return DefaultRule()
                end
            else
                if all(i -> i in ix, iy)
                    return Sum()
                elseif all(i -> i in iy, ix)  # e.g. ij->ijk
                    return Repeat()
                else  # e.g. ijkxc,ijkl
                    return DefaultRule()
                end
            end
        else  # ix is not unique
            if all(i -> i in ix, iy) && all(i -> i in iy, ix)   # ijjj->ij
                return Diag()
            else
                return DefaultRule()
            end
        end
    else  # iy is not unique
        if allunique(ix) && all(x->x∈iy, ix)
            if all(y->y∈ix, iy)  # e.g. ij->ijjj
                return Duplicate()
            else  # e.g. ij->ijjl
                return DefaultRule()
            end
        else
            return DefaultRule()
        end
    end
end

match_rule(code::EinCode) = match_rule(getixs(code), getiy(code))


@inline function _add_batch(::SimpleBinaryRule{ix1,ix2,iy}) where {ix1,ix2,iy}
    SimpleBinaryRule{(ix1...,'l'), (ix2...,'l'), (iy...,'l')}()
end
@inline _add_batch(::DefaultRule) = DefaultRule()

function match_rule_binary(ix1, ix2, iy)
    Nx1, Nx2, Ny = length(ix1), length(ix2), length(iy)
    if !_isunique(ix1) || !_isunique(ix2) || !_isunique(iy)
        DefaultRule()
    elseif (Nx1 + Nx2 + Ny) % 2 == 0 # no batch
        _match_simple2(ix1,ix2,iy,Nx1,Nx2,Ny)
    elseif Nx1>0 && Nx2>0 && Ny>0 && ix1[Nx1]==ix2[Nx2]==iy[Ny]
        rule = _match_simple2(ix1,ix2,iy,Nx1-1,Nx2-1,Ny-1)
        _add_batch(rule)
    else
        DefaultRule()
    end
end
@inline function _isunique(ix)
    if length(ix) <= 1
        return true
    elseif length(ix) == 2
        return @inbounds ix[1] != ix[2]
    elseif length(ix) == 3
        @inbounds a, b, c = ix
        return a != c && a != c && a != b
    else  # to default rules
        return false
    end
end

function _match_simple2(ix1, ix2, iy, Nx1, Nx2, Ny)
    if Nx1==0
        if (Ny==Nx2==0)
            return SimpleBinaryRule((), (), ())
        elseif (Ny==Nx2==1 && ix2[1] == iy[1])
            return SimpleBinaryRule((), ('k',), ('k',))
        end
    elseif Nx1==1
        if (Nx2==0 && Ny==1 && iy[1]==ix1[1])
            return SimpleBinaryRule(('i',), (), ('i',))
        elseif (Nx2==1 && Ny==0 && ix1[1]==ix2[1])
            return SimpleBinaryRule(('j',), ('j',), ())
        elseif Nx2==1 && Ny==2
            if (iy[1]==ix1[1] && iy[2]==ix2[1])
                return SimpleBinaryRule(('i',), ('k',), ('i','k'))
            elseif iy[1]==ix2[1] && iy[2]==ix1[1]
                return SimpleBinaryRule(('i',), ('k',), ('k','i'))
            end
        elseif Nx2==2 && Ny==1
            if ix2[1]==ix1[1] && ix2[2]==iy[1]
                return SimpleBinaryRule(('j',), ('j','k'), ('k',))
            elseif ix2[1]==iy[1] && ix2[2]==ix1[1]
                return SimpleBinaryRule(('j',), ('k','j'), ('k',))
            end
        end
    elseif Nx1==2
        if Nx2==1 && Ny==1
            if ix1[1]==ix2[1] && ix1[2]==iy[1]
                return SimpleBinaryRule(('j','i'), ('j',), ('i',))
            elseif ix1[1]==iy[1] && ix1[2]==ix2[1]
                return SimpleBinaryRule(('i','j'), ('j',), ('i',))
            end
        elseif (Nx2==2 && Ny==2)
            if ix1[1]==ix2[1] && ix1[2]==iy[1] && ix2[2]==iy[2]
                return SimpleBinaryRule(('j','i'), ('j','k'), ('i','k'))
            elseif ix1[1]==ix2[2] && ix1[2]==iy[1] && ix2[1]==iy[2]
                return SimpleBinaryRule(('j','i'), ('k','j'), ('i','k'))
            elseif ix1[1]==ix2[1] && ix1[2]==iy[2] && ix2[2]==iy[1]
                return SimpleBinaryRule(('j','i'), ('j','k'), ('k','i'))
            elseif ix1[1]==ix2[2] && ix1[2]==iy[2] && ix2[1]==iy[1]
                return SimpleBinaryRule(('j','i'), ('k','j'), ('k','i'))
            elseif ix1[2]==ix2[1] && ix1[1]==iy[1] && ix2[2]==iy[2]
                return SimpleBinaryRule(('i','j'), ('j','k'), ('i','k'))
            elseif ix1[2]==ix2[2] && ix1[1]==iy[1] && ix2[1]==iy[2]
                return SimpleBinaryRule(('i','j'), ('k','j'), ('i','k'))
            elseif ix1[2]==ix2[1] && ix1[1]==iy[2] && ix2[2]==iy[1]
                return SimpleBinaryRule(('i','j'), ('j','k'), ('k','i'))
            elseif ix1[2]==ix2[2] && ix1[1]==iy[2] && ix2[1]==iy[1]
                return SimpleBinaryRule(('i','j'), ('k','j'), ('k','i'))
            end
        end
    end
    return DefaultRule()
end

function simplify_unary(ix::Vector{T}, iy::Vector{T}, x, size_dict::Dict{T}) where T
    if ix == iy
        return x
    elseif length(ix) == length(iy) # permutation
        return einsum(Permutedims(), (ix,), iy, (x,), size_dict)
    else
        # diag
        ix_ = unique(ix)
        x_ = length(ix_) != length(ix) ? einsum(Diag(), (ix,), ix_, (x,), size_dict) : x
        # sum
        if length(ix_) != length(iy)
            return einsum(Sum(), (ix_,), iy, (x_,), size_dict)
        elseif ix_ != iy
            return einsum(Permutedims(), (ix_,), iy, (x_,), size_dict)
        else
            return x_
        end
    end
end

function expand_unary(ix::Vector{T}, iy::Vector{T}, x::AbstractArray, size_dict::Dict{T}) where T
    iy_b = unique(iy)
    iy_a = filter(i->i ∈ ix, iy_b)
    y_a = if ix != iy_a
        einsum(Permutedims(), (ix,), iy_a, (x,), size_dict)
    else
        x
    end
    # repeat
    y_b = length(iy_a) != length(iy_b) ? einsum(Repeat(), (iy_a,), iy_b, (y_a,), size_dict) : y_a
    # duplicate
    length(iy_b) != length(iy) ? einsum(Duplicate(), (iy_b,), iy, (y_b,), size_dict) : y_b
end

"""
Get the expected labels.
"""
function analyze_binary(ix1::Vector{T}, ix2::Vector{T}, iy::Vector{T}, size_dict::Dict{T,Int}) where T
    ix_inner, ix1_outer, ix2_outer, batch = _analyze_binary_input(ix1, ix2, iy)
    c1 = vcat(ix1_outer, ix_inner, batch)
    c2 = vcat(ix_inner, ix2_outer, batch)
    cy = vcat(ix1_outer, ix2_outer, batch)
    si = prod(map(x->size_dict[x], ix1_outer))
    sj = prod(map(x->size_dict[x], ix_inner))
    sk = prod(map(x->size_dict[x], ix2_outer))
    sl = prod(map(x->size_dict[x], batch))
    has_i = !isempty(ix1_outer)
    has_j = !isempty(ix_inner)
    has_k = !isempty(ix2_outer)
    has_l = !isempty(batch)
    i1 = Char[]
    i2 = Char[]
    iyb = Char[]
    s1 = Int[]
    s2 = Int[]
    if has_i
        push!(i1, 'i')
        push!(iyb, 'i')
        push!(s1, si)
    end
    if has_j
        push!(i1, 'j')
        push!(i2, 'j')
        push!(s1, sj)
        push!(s2, sj)
    end
    if has_k
        push!(i2, 'k')
        push!(iyb, 'k')
        push!(s2, sk)
    end
    if has_l
        push!(i1, 'l')
        push!(i2, 'l')
        push!(iyb, 'l')
        push!(s1, sl)
        push!(s2, sl)
    end
    return c1, c2, cy, s1, s2, i1, i2, iyb
end

function _analyze_binary_input(ix1::Vector{T}, ix2::Vector{T}, iy::Vector{T}) where T
    ix1_batch = T[]
    ix1_inner = T[]
    ix1_outer = T[]
    for l1 in ix1
        if l1 ∈ ix2
            if l1 ∈ iy  # batch
                l1 ∉ ix1_batch && push!(ix1_batch, l1)
            else        # inner
                l1 ∉ ix1_inner && push!(ix1_inner, l1)
            end
        elseif l1 ∈ iy  # outer
            l1 ∉ ix1_outer && push!(ix1_outer, l1)
        else
            # dangling
        end
    end
    ix2_outer = T[]     # outer dimension of x2
    for l2 in ix2
        if l2 ∉ ix1 && l2 ∈ iy && l2 ∉ ix2_outer
            push!(ix2_outer, l2)
        end
    end
    ix1_inner, ix1_outer, ix2_outer, ix1_batch
end