# Code is a binary representation of `(O1,I,O2,B)`.
# Because the time complexity of `MatMul` and `BatchedMatMul` are higher than space complexity, we allow `permutedims`.
# We reduce the contraction to these basic forms through `permutedims` and reshape,
# because we assume most using cases have both inner and outer degrees on freedom.

# ,-> : 000
# S = 1
# T = 1
function einsum(code::EinCode{((),()), ()}, xs::NTuple{2, Any}, size_dict)
    asarray(xs[1][] * xs[2])
end

# i,->i : 100
# S = N
# T = N
function einsum(code::EinCode{(('i',),()), ('i',)}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* xs[2][]
end

# j,j-> : 010
# S = N
# T = N
function einsum(code::EinCode{(('j',), ('j',)), ()}, xs::NTuple{2, Any}, size_dict)
    asarray(transpose(xs[1]) * xs[2])
end

# ,k->k : 001
# S = N
# T = N
@inline function einsum(code::EinCode{((), ('k',)), ('k',)}, xs::NTuple{2, Any}, size_dict)
    einsum(EinCode{(('i',),()),('i',)}(), (xs[2], xs[1]), size_dict)
end

# j,jk->k : 011
# S = N^2
# T = N^2
function einsum(code::EinCode{(('j',), ('j','k')), ('k',)}, xs::NTuple{2, Any}, size_dict)
    vec(transpose(xs[1]) * xs[2])
end
function einsum(code::EinCode{(('j',), ('k','j')), ('k',)}, xs::NTuple{2, Any}, size_dict)
    xs[2] * xs[1]
end

# ij,j->i : 110
# S = N^2
# T = N^2
@inline function einsum(code::EinCode{(('i','j'),('j',)), ('i',)}, xs::NTuple{2, Any}, size_dict)
    einsum(EinCode{(('j',),('k','j')), ('k',)}(), (xs[2], xs[1]), size_dict)
end
@inline function einsum(code::EinCode{(('j','i'),('j',)), ('i',)}, xs::NTuple{2, Any}, size_dict)
    einsum(EinCode{(('j',),('j','k')), ('k',)}(), (xs[2], xs[1]), size_dict)
end

# i,k->ik : 101
# S = N^2
# T = N^2
function einsum(code::EinCode{(('i',), ('k',)), ('i','k')}, xs::NTuple{2, Any}, size_dict)
    xs[1] * transpose(xs[2])
end
@inline function einsum(code::EinCode{(('i',), ('k',)), ('k','i')}, xs::NTuple{2, Any}, size_dict)
    einsum(EinCode{(('i',),('k',)),('i','k')}(), (xs[2], xs[1]), size_dict)
end

# 000
function einsum(code::EinCode{(('l',),('l',)), ('l',)}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* xs[2]
end

# 100
function einsum(code::EinCode{(('i','l'),('l',)), ('i','l')}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* transpose(xs[2])
end

# 001
@inline function einsum(code::EinCode{(('l',), ('k','l')), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    einsum(EinCode{(('i','l'),('l',)),('i','l')}(), (xs[2], xs[1]), size_dict)
end

# 010
function einsum(code::EinCode{(('j','l'), ('j','l')), ('l')}, xs::NTuple{2, Any}, size_dict)
    a, b = xs
    T = promote_type(T1, T2)
    out = similar(a, T, size(a, 2))
    @inbounds for k=1:size(a, 2)
        elem = zero(T)
        for i=1:size(a, 1)
            elem += a[i,k] * b[i,k]
        end
        a[k] = elem
    end
    return out
end

# 101
function einsum(code::EinCode{(('i','l'), ('k','l')), ('i','k','l')}, xs::NTuple{2, Any}, size_dict)
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
@inline function einsum(code::EinCode{(('i','l'), ('k','l')), ('k','i','l')}, xs::NTuple{2, Any}, size_dict)
    einsum(EinCode{(('i','l'),('k','l')), ('i','k','l')}(), (xs[2], xs[1]), size_dict)
end

# 011
function einsum(code::EinCode{(('j','l'), ('j','k','l')), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    loop_einsum(code, xs, size_dict)
end
function einsum(code::EinCode{(('j','l'), ('k','j','l')), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    loop_einsum(code, xs, size_dict)
end

# 110
@inline function einsum(code::EinCode{(('i','j','l'), ('j','l')), ('i','l')}, xs::NTuple{2, Any}, size_dict)
    einsum(EinCode{(('j','l'), ('k','j','l')), ('k','l')}(), (xs[2],xs[1]), size_dict)
end
@inline function einsum(code::EinCode{(('j','i','l'), ('j','l')), ('i','l')}, xs::NTuple{2, Any}, size_dict)
    einsum(EinCode{(('j','l'), ('j','k','l')), ('k','l')}(), (xs[2],xs[1]), size_dict)
end

# ij,jk->ik : 111
# S = N^2
# T = N^3
for (i1, X1) in enumerate([('i', 'j'), ('j', 'i')])
    for (i2, X2) in enumerate([('j', 'k'), ('k', 'j')])
        for (i3, X3) in enumerate([('i', 'k'), ('k', 'i')])
            A1 = i1==i3 ? :(xs[1]) : :(transpose(xs[1]))
            A2 = i2==i3 ? :(xs[2]) : :(transpose(xs[2]))
            @eval function einsum(code::EinCode{($X1,$X2), $X3}, xs::NTuple{2, Any}, size_dict)
                $(i3==1 ? :($A1*$A2) : :($A2*$A1))
            end
            X1B = (X1...,'l')
            X2B = (X2...,'l')
            X3B = (X3...,'l')
            @eval function einsum(code::EinCode{($X1B,$X2B), $X3B}, xs::NTuple{2, Any}, size_dict)
                loop_einsum(code, xs, size_dict)
            end
            C1 = i1==i3 ? 'N' : 'T'
            C2 = i2==i3 ? 'N' : 'T'
            @eval function einsum(::EinCode{($X1B,$X2B),$X3B}, xs::NTuple{2, AbstractArray{<:BlasFloat}}, size_dict)
                $(i3==1 ? :(batched_gemm($C1, $C2, xs[1], xs[2])) : :(batched_gemm($C2, $C1, xs[2], xs[1])))
            end
        end
    end
end

function preprocess_binary(ix1, ix2, iy, x1, x2, size_dict)
    c1, c2, cy, s1, s2, sy, code = analyze_binary(ix1, ix2, iy, size_dict)
    x1_ = reshape(einsum(EinCode{(ix1,), c1}, x1, size_dict), s1)
    x2_ = reshape(einsum(EinCode{(ix2,), c2}, x2, size_dict), s2)
    y_ = reshape(einsum(code, (x1_, x2_), IndexSize(('i', 'j', 'k', 'l'), (si, sj, sk, sl))), sy)
    return einsum(EinCode{((cy,),), iy}(), y_, size_dict)
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
    si = prod(x->size_dict[x], ix1_outer; init=1)
    sj = prod(x->size_dict[x], ix_inner; init=1)
    sk = prod(x->size_dict[x], ix2_outer; init=1)
    sl = prod(x->size_dict[x], batch; init=1)
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
    code = EinCode{((i1...,), (i2...,)), (iy...,)}()
    return c1, c2, cy, (s1...,), (s2...,), sy, code
end

function _analyze_binary_input(ix1, ix2, iy)
    T = eltype(ix1)
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