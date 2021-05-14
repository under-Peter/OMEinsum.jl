# binary operations that can not be simplified by a unitary operations
struct SimpleBinaryRule{ix1,ix2,iy} <: EinRule{2} end
function SimpleBinaryRule(code::EinCode{ixs,iy}) where {ixs, iy}
    @assert length(ixs)==2 "fail to construct simple binary rule from $code"
    SimpleBinaryRule{ixs[1], ixs[2], iy}()
end
SimpleBinaryRule(ix1, ix2, iy) = SimpleBinaryRule{ix1, ix2, iy}()

@inline function match_rule(ixs::Tuple{NTuple{Nx1,T},NTuple{Nx2,T}}, iy::NTuple{Ny,T}) where {Nx1,Nx2,Ny,T}
    match_simplebinary(ixs..., iy)
end

@inline function _add_patch(::SimpleBinaryRule{ix1,ix2,iy}) where {ix1,ix2,iy}
    SimpleBinaryRule{(ix1...,'l'), (ix2...,'l'), (iy...,'l')}()
end
@inline _add_patch(::DefaultRule) = DefaultRule()

function match_simplebinary(ix1, ix2, iy)
    Nx1, Nx2, Ny = length(ix1), length(ix2), length(iy)
    if (Nx1 + Nx2 + Ny) % 2 == 0 # no batch
        _match_simple2(ix1,ix2,iy,Nx1,Nx2,Ny)
    elseif Nx1>0 && Nx2>0 && Ny>0 && ix1[Nx1]==ix2[Nx2]==iy[Ny]
        rule = _match_simple2(ix1,ix2,iy,Nx1-1,Nx2-1,Ny-1)
        _add_patch(rule)
    else
        DefaultRule()
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

function einsum(rule::SimpleBinaryRule, ::EinCode{ixs, iy}, xs::NTuple{2, Any}, size_dict) where {ixs, iy}
    @debug rule ixs => iy size.(xs)
    einsum(rule, xs, size_dict)
end
# Code is a binary representation of `(O1,I,O2,B)`.
# Because the time complexity of `GEMM` and `BatchedGEMM` are higher than space complexity, we allow `permutedims`.
# We reduce the contraction to these basic forms through `permutedims` and reshape,
# because we assume most using cases have both inner and outer degrees on freedom.

# ,-> : 000
# S = 1
# T = 1
function einsum(::SimpleBinaryRule{(),(), ()}, xs::NTuple{2, Any}, size_dict)
    asarray(xs[1][] * xs[2], xs[1])
end

# i,->i : 100
# S = N
# T = N
function einsum(::SimpleBinaryRule{('i',),(), ('i',)}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* xs[2][]
end

# j,j-> : 010
# S = N
# T = N
function einsum(::SimpleBinaryRule{('j',), ('j',), ()}, xs::NTuple{2, Any}, size_dict)
    asarray(transpose(xs[1]) * xs[2], xs[1])
end

# ,k->k : 001
# S = N
# T = N
@inline function einsum(::SimpleBinaryRule{(), ('k',), ('k',)}, xs::NTuple{2, Any}, size_dict)
    einsum(SimpleBinaryRule{('i',),(),('i',)}(), (xs[2], xs[1]), size_dict)
end

# j,jk->k : 011
# S = N^2
# T = N^2
function einsum(::SimpleBinaryRule{('j',), ('j','k'), ('k',)}, xs::NTuple{2, Any}, size_dict)
    vec(transpose(xs[1]) * xs[2])
end
function einsum(::SimpleBinaryRule{('j',), ('k','j'), ('k',)}, xs::NTuple{2, Any}, size_dict)
    xs[2] * xs[1]
end

# ij,j->i : 110
# S = N^2
# T = N^2
@inline function einsum(::SimpleBinaryRule{('i','j'),('j',), ('i',)}, xs::NTuple{2, Any}, size_dict)
    einsum(SimpleBinaryRule{('j',),('k','j'), ('k',)}(), (xs[2], xs[1]), size_dict)
end
@inline function einsum(::SimpleBinaryRule{('j','i'),('j',), ('i',)}, xs::NTuple{2, Any}, size_dict)
    einsum(SimpleBinaryRule{('j',),('j','k'), ('k',)}(), (xs[2], xs[1]), size_dict)
end

# i,k->ik : 101
# S = N^2
# T = N^2
function einsum(::SimpleBinaryRule{('i',), ('k',), ('i','k')}, xs::NTuple{2, Any})
    xs[1] * transpose(xs[2])
end
@inline function einsum(::SimpleBinaryRule{('i',), ('k',),('k','i')}, xs::NTuple{2, Any})
    einsum(SimpleBinaryRule{('i',),('k',),('i','k')}(), (xs[2], xs[1]), size_dict)
end

# 000
function einsum(::SimpleBinaryRule{('l',),('l',), ('l',)}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* xs[2]
end

# 100
function einsum(::SimpleBinaryRule{('i','l'),('l',), ('i','l')}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* transpose(xs[2])
end

# 001
@inline function einsum(::SimpleBinaryRule{('l',), ('k','l'), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    einsum(SimpleBinaryRule{('i','l'),('l',),('i','l')}(), (xs[2], xs[1]), size_dict)
end

# 010
function einsum(::SimpleBinaryRule{('j','l'), ('j','l'), ('l',)}, xs::NTuple{2, Any}, size_dict)
    a, b = xs
    T = promote_type(eltype(xs[1]), eltype(xs[2]))
    out = similar(a, T, size(a, 2))
    @inbounds for k=1:size(a, 2)
        elem = zero(T)
        for i=1:size(a, 1)
            elem += a[i,k] * b[i,k]
        end
        out[k] = elem
    end
    return out
end

# 101
function einsum(::SimpleBinaryRule{('i','l'), ('k','l'), ('i','k','l')}, xs::NTuple{2, Any}, size_dict)
    a, b = xs
    T = promote_type(eltype(xs[1]), eltype(xs[2]))
    out = similar(a, T, size(a, 1), size(b, 1), size(a, 2))
    @inbounds for k=1:size(a, 2)
        for j=1:size(b, 1)
            for i=1:size(a, 1)
                out[i,j,k] = a[i,k] * b[j,k]
            end
        end
    end
    return out
end
@inline function einsum(::SimpleBinaryRule{('i','l'), ('k','l'), ('k','i','l')}, xs::NTuple{2, Any}, size_dict)
    einsum(SimpleBinaryRule{('i','l'),('k','l'), ('i','k','l')}(), (xs[2], xs[1]))
end

# 011
function einsum(::SimpleBinaryRule{('j','l'), ('j','k','l'), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    size_dict = IndexSize(('j','k','l'), size(xs[2]))
    loop_einsum(EinCode{(('j','l'), ('j','k','l')), ('k','l')}(), xs, size_dict)
end
function einsum(::SimpleBinaryRule{('j','l'), ('k','j','l'), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    size_dict = IndexSize(('j','k','l'), size(xs[2]))
    loop_einsum(EinCode{(('j','l'), ('k','j','l')), ('k','l')}(), xs, size_dict)
end

# 110
@inline function einsum(::SimpleBinaryRule{('i','j','l'), ('j','l'), ('i','l')}, xs::NTuple{2, Any}, size_dict)
    einsum(SimpleBinaryRule{('j','l'), ('k','j','l'), ('k','l')}(), (xs[2],xs[1]))
end
@inline function einsum(::SimpleBinaryRule{('j','i','l'), ('j','l'), ('i','l')}, xs::NTuple{2, Any}, size_dict)
    einsum(SimpleBinaryRule{('j','l'), ('j','k','l'), ('k','l')}(), (xs[2],xs[1]), size_dict)
end

# ij,jk->ik : 111
# S = N^2
# T = N^3
for (i1, X1) in enumerate([('i', 'j'), ('j', 'i')])
    for (i2, X2) in enumerate([('j', 'k'), ('k', 'j')])
        for (i3, X3) in enumerate([('i', 'k'), ('k', 'i')])
            A1 = i1==i3 ? :(xs[1]) : :(transpose(xs[1]))
            A2 = i2==i3 ? :(xs[2]) : :(transpose(xs[2]))
            @eval function einsum(::SimpleBinaryRule{$X1,$X2, $X3}, xs::NTuple{2, Any}, size_dict)
                $(i3==1 ? :($A1*$A2) : :($A2*$A1))
            end
            X1B = (X1...,'l')
            X2B = (X2...,'l')
            X3B = (X3...,'l')
            @eval function einsum(::SimpleBinaryRule{$X1B,$X2B, $X3B}, xs::NTuple{2, Any}, size_dict)
                size_dict = IndexSize(('i', 'j', 'k', 'l'), (size(xs[1], 1), size(xs[2])...))
                loop_einsum(EinCode{($X1B,$X2B), $X3B}(), xs, size_dict)
            end
            C1 = i1==i3 ? 'N' : 'T'
            C2 = i2==i3 ? 'N' : 'T'
            @eval function einsum(::SimpleBinaryRule{$X1B,$X2B,$X3B}, xs::NTuple{2, AbstractArray{<:BlasFloat}}, size_dict)
                $(i3==1 ? :(_batched_gemm($C1, $C2, xs[1], xs[2])) : :(_batched_gemm($C2, $C1, xs[2], xs[1])))
            end
        end
    end
end

function einsum(::DefaultRule, ::EinCode{ixs, iy}, xs::NTuple{2, Any}, size_dict) where {ixs, iy}
    @debug "DefaultRule binary" ixs => iy size.(xs)
    ix1, ix2 = ixs
    x1, x2 = xs
    c1, c2, cy, s1, s2, sy, rule, sdict = analyze_binary(ix1, ix2, iy, size_dict)
    if ix1 !== c1
        if length(ix1) == length(c1) # permutation
            x1 = einsum(Permutedims(), EinCode{(ix1,), c1}(), (x1,), size_dict)
        else
            x1 = einsum(EinCode{(ix1,), c1}(), (x1,), size_dict)
        end
    end
    if ix2 !== c2
        if length(ix2) == length(c2) # permutation
            x2 = einsum(Permutedims(), EinCode{(ix2,), c2}(), (x2,), size_dict)
        else
            x2 = einsum(EinCode{(ix2,), c2}(), (x2,), size_dict)
        end
    end
    x1_ = reshape(x1, s1)
    x2_ = reshape(x2, s2)
    y_ = reshape(einsum(rule, (x1_, x2_), sdict), sy)
    return einsum(EinCode{(cy,), iy}(), (y_,), size_dict)
end

"""
Get the expected labels.
"""
function analyze_binary(ix1, ix2, iy, size_dict)
    ix_inner, ix1_outer, ix2_outer, batch = _analyze_binary_input(ix1, ix2, iy)
    c1 = (ix1_outer..., ix_inner..., batch...)
    c2 = (ix_inner..., ix2_outer..., batch...)
    cy = (ix1_outer..., ix2_outer..., batch...)
    sy = map(x->size_dict[x], cy)
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
    iy = Char[]
    s1 = Int[]
    s2 = Int[]
    if has_i
        push!(i1, 'i')
        push!(iy, 'i')
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
        push!(iy, 'k')
        push!(s2, sk)
    end
    if has_l
        push!(i1, 'l')
        push!(i2, 'l')
        push!(iy, 'l')
        push!(s1, sl)
        push!(s2, sl)
    end
    rule = SimpleBinaryRule{(i1...,), (i2...,), (iy...,)}()
    sdict = IndexSize('i'=>si, 'j'=>sj, 'k'=>sk, 'l'=>sl)
    return c1, c2, cy, (s1...,), (s2...,), sy, rule, sdict
end

function _analyze_binary_input(ix1::NTuple{N1,T}, ix2::NTuple{N2,T}, iy::NTuple{Ny,T}) where {T,N1,N2,Ny}
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