# Code is a binary representation of `(O1,I,O2,B)`.
struct LeafCode{I}
end

# Because the time complexity of `MatMul` and `BatchedMatMul` are higher than space complexity, we allow `permutedims`.
# We reduce the contraction to these basic forms through `permutedims` and reshape,
# because we assume most using cases have both inner and outer degrees on freedom.
for (ID, CT, N1, N2, N2) in [
    (0, :ScalarProduct, 0, 0, 0), (4, :VecScalar, 1, 0, 1),
    (1, :ScalarVec, 0, 1, 1), (2, :InnerProduct, 1, 1, 0),
    (5, :OuterProduct, 1, 1, 2), (7, :MatMul, 2, 2, 2),
    (6, :MatVec, 2, 1, 1), (3, :VecMat, 1, 2, 1),
    ]
    @eval const $CT = LeafCode{$(ID*2)}
    @eval function match_simplerule(ix1::NTuple{$N1}, ix2::NTuple{$N2}, iy::NTuple{$N3})
        $CT()
    end
    BCT = Symbol(:Batched, CT)
    @eval const $BCT = LeafCode{$(ID*2+1)}
    @eval function match_simplerule(ix1::NTuple{$(N1+1)}, ix2::NTuple{$(N2+1)}, iy::NTuple{$(N3+1)})
        $CT()
    end
end

# ,-> : 000
# S = 1
# T = 1
function einsum(code::EinCode{((),()), ()}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* xs[2]
end

# i,->i : 100
# S = N
# T = N
function einsum(code::EinCode{(('i',),()), ('i',)}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* xs[2][]
end

# ,k->k : 001
# S = N
# T = N
function einsum(code::EinCode{((), ('k',)), ('k',)}, xs::NTuple{2, Any}, size_dict)
    xs[1][] .* xs[2]
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
function einsum(code::EinCode{(('i','j'),('j')), ('i',)}, xs::NTuple{2, Any}, size_dict)
    xs[1] * xs[2]
end
function einsum(code::EinCode{(('j','i'),('j')), ('i',)}, xs::NTuple{2, Any}, size_dict)
    vec(transpose(xs[2]) * xs[1])
end

# j,j-> : 010
# S = N
# T = N
function einsum(code::EinCode{(('j',), ('j',)), ()}, xs::NTuple{2, Any}, size_dict)
    transpose(xs[1]) * xs[2]
end

# i,k->ik : 101
# S = N^2
# T = N^2
function einsum(code::EinCode{(('i',), ('k',)), ('i','k')}, xs::NTuple{2, Any}, size_dict)
    xs[1] * transpose(xs[2])
end
function einsum(code::EinCode{(('i',), ('k',)), ('k','i')}, xs::NTuple{2, Any}, size_dict)
    transpose(xs[2]) * xs[1]
end

# ij,jk->ik : 111
# S = N^2
# T = N^3
function einsum(code::EinCode{(('i','j'),('j','k')), ('i','k')}, xs::NTuple{2, Any}, size_dict)
    xs[1] * xs[2]
end
function einsum(code::EinCode{(('i','j'),('j','k')), ('k','i')}, xs::NTuple{2, Any}, size_dict)
    transpose(xs[2]) * transpose(xs[1])
end
function einsum(code::EinCode{(('j','i'),('j','k')), ('i','k')}, xs::NTuple{2, Any}, size_dict)
    transpose(xs[1]) * xs[2]
end
function einsum(code::EinCode{(('j','i'),('j','k')), ('k','i')}, xs::NTuple{2, Any}, size_dict)
    transpose(xs[2]) * xs[1]
end
function einsum(code::EinCode{(('i','j'),('k','j')), ('i','k')}, xs::NTuple{2, Any}, size_dict)
    xs[1] * transpose(xs[2])
end
function einsum(code::EinCode{(('i','j'),('k','j')), ('k','i')}, xs::NTuple{2, Any}, size_dict)
    xs[2] * transpose(xs[1])
end
function einsum(code::EinCode{(('j','i'),('k','j')), ('i','k')}, xs::NTuple{2, Any}, size_dict)
    transpose(xs[1]) * transpose(xs[2])
end
function einsum(code::EinCode{(('j','i'),('k','j')), ('k','i')}, xs::NTuple{2, Any}, size_dict)
    xs[2] * xs[1]
end

function einsum(code::EinCode{(('l',),('l',)), ('l',)}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* xs[2]
end

function einsum(code::EinCode{(('i','l'),('l',)), ('i','l')}, xs::NTuple{2, Any}, size_dict)
    xs[1] .* transpose(xs[2])
end

function einsum(code::EinCode{(('l',), ('k','l')), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    transpose(xs[1]) .* xs[2]
end

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

function einsum(code::EinCode{(('i','l'), ('k','l')), ('i','k','l')}, xs::NTuple{2, Any}, size_dict)
    T = promote_type(T1, T2)
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
    einsum(EinCode{(('i','l'), ('k','l')), ('i','k','l')}(), (xs[2], xs[1]), size_dict)
end

function einsum(code::EinCode{(('j','l'), ('j','k','l')), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    loop_einsum(code, xs, size_dict)
end
function einsum(code::EinCode{(('j','l'), ('k','j','l')), ('k','l')}, xs::NTuple{2, Any}, size_dict)
    loop_einsum(code, xs, size_dict)
end

function einsum(code::EinCode{(('i','j','l'), ('j','l')), ('i','l')}, xs::NTuple{2, Any}, size_dict) where {ixs, iy}
    loop_einsum(code, xs, size_dict)
end
function einsum(code::EinCode{(('j','i','l'), ('j','l')), ('i','l')}, xs::NTuple{2, Any}, size_dict) where {ixs, iy}
    loop_einsum(code, xs, size_dict)
end

function einsum(code::EinCode{ixs, iy}, xs::NTuple{2, Any}, size_dict) where {ixs, iy}
    loop_einsum(code, xs, size_dict)
end

function einsum(::EinCode{ixs,iy}, xs::NTuple{2, AbstractArray{<:BlasFloat}}, size_dict) where {ixs, iy}
    batched_contract(Val(ixs[1]), xs[1], Val(ixs[2]), xs[2], Val(iy))
end

function proprocess_input(::EinCode{ixs,iy}, idx) where {ixs, iy}
    _preprocess_dupindices(ixs)
end

@generated function _preprocess_dupindices(::Val{ix}, x) where {ix}
    if length(tunique(ix)) != length(ix)
        iy = [l for l in ix if count(==(l), ix) == 1]
        :(($(Val((iy...,))), einsum($(EinCode((ix,), (iy...,))), (x,), get_size_dict(($ix,), (x,)))))
    else
        :(($(Val(ix)), x))
    end
end

"""
Get the expected labels.
"""
function analyze_binary(ix1, ix2, iy, size_dict)
    ix_inner, ix1_outer, ix2_outer, batch = _analyze_binary_input(ix1, ix2, iy)
    s1 = (prod(x->size_dict[x], ix1_outer; init=1), prod(x->size_dict[x], ix_inner; init=1), prod(x->size_dict[x], batch; init=1))
    s2 = (prod(x->size_dict[x], ix_inner; init=1), prod(x->size_dict[x], ix2_outer; init=1), prod(x->size_dict[x], batch; init=1))
    sy = map(x->size_dict[x], iy)
    #c1 = EinCode{(ix1,), (ix1_outer..., ix_inner..., batch...)}()
    #c2 = EinCode{(ix2,), (ix_inner..., ix2_outer..., batch...)}()
    #cy = EinCode{((ix1_outer..., ix2_outer..., batch...),), iy}()
    c1 = (ix1_outer..., ix_inner..., batch...)
    c2 = (ix_inner..., ix2_outer..., batch...)
    cy = (ix1_outer..., ix2_outer..., batch...)
    t1 = (!isempty(ix1_outer) ? 4 : 0) + (!isempty(ix_inner) ? 8 : 0) + (!isempty(ix2_outer) ? 2 : 0) + (!isempty(batch) ? 1 : 0)
    return c1, c2, cy, s1, s2, sy
end

function preprocess_binary(ix1, ix2, iy, x1, x2, size_dict)
    c1, c2, cy, s1, s2, sy = analyze_binary(ix1, ix2, iy, size_dict)
    x1_ = reshape(EinCode{(ix1,), c1}(x1), s1)
    x2_ = reshape(EinCode{(ix2,), c2}(x2), s1)
    y_ = EinCode{(c1,c2), cy}(x1_, x2_)
    y = EinCode{(cy,), iy}(y_)
    return reshape(y, sy)
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

@testset "analyse binary" begin
    size_dict = OMEinsum.IndexSize(1=>1, 2=>2, 3=>3, 4=>4, 6=>6, 7=>7)
    c1, c2, cy, s1, s2, sy = analyze_binary((1,2,3,4), (2,6,6,4,2), (7,2,1,2,2,6), size_dict)
    @test c1 == (1,4,2)
    @test c2 == (4,6,2)
    @test cy == (1,6,2)
    @test s1 == (1,4,2)
    @test s2 == (4,6,2)
    @test sy == (7,2,1,2,2,6)
end