# binary operations that can not be simplified by a unitary operations
struct SimpleBinaryRule{ix1,ix2,iy} <: EinRule{2} end
function SimpleBinaryRule(code::EinCode)
    ixs, iy = getixs(code), getiy(code)
    @assert length(ixs)==2 "fail to construct simple binary rule from $code"
    SimpleBinaryRule{ixs[1], ixs[2], iy}()
end
SimpleBinaryRule(ix1, ix2, iy) = SimpleBinaryRule{ix1, ix2, iy}()

# Code is a binary representation of `(O1,I,O2,B)`.
# Because the time complexity of `GEMM` and `BatchedGEMM` are higher than space complexity, we allow `permutedims`.
# We reduce the contraction to these basic forms through `permutedims` and reshape,
# because we assume most using cases have both inner and outer degrees on freedom.

# ,-> : 000
# S = 1
# T = 1
function binary_einsum!(::SimpleBinaryRule{(),(), ()}, x1, x2, y, sx, sy)
    @addmul! sy * y + sx * x1 * x2
end

# i,->i : 100
# S = N
# T = N
function binary_einsum!(::SimpleBinaryRule{('i',),(), ('i',)}, x1, x2, y, sx, sy)
    @addmul! sy * y + sx * x1 * Ref(asscalar(x2))
end

# j,j-> : 010
# S = N
# T = N
function binary_einsum!(::SimpleBinaryRule{('j',), ('j',), ()}, x1, x2, y, sx, sy)
    @addmul! sy * y + sx * Ref(transpose(x1) * x2)
end

# ,k->k : 001
# S = N
# T = N
@inline function binary_einsum!(::SimpleBinaryRule{(), ('k',), ('k',)}, x1, x2, y, sx, sy)
    binary_einsum!(SimpleBinaryRule{('i',),(),('i',)}(), x2, x1, y, sx, sy)
end

# j,jk->k : 011
# S = N^2
# T = N^2
function binary_einsum!(::SimpleBinaryRule{('j',), ('j','k'), ('k',)}, x1, x2, y, sx, sy)
    mul!(y, transpose(x2), x1, sx, sy)
end
function binary_einsum!(::SimpleBinaryRule{('j',), ('k','j'), ('k',)}, x1, x2, y, sx, sy)
    mul!(y, x2, x1, sx, sy)
end

# ij,j->i : 110
# S = N^2
# T = N^2
@inline function binary_einsum!(::SimpleBinaryRule{('i','j'),('j',), ('i',)}, x1, x2, y, sx, sy)
    mul!(y, x1, x2, sx, sy)
end
@inline function binary_einsum!(::SimpleBinaryRule{('j','i'),('j',), ('i',)}, x1, x2, y, sx, sy)
    mul!(y, transpose(x1), x2, sx, sy)
end

# i,k->ik : 101
# S = N^2
# T = N^2
function binary_einsum!(::SimpleBinaryRule{('i',), ('k',), ('i','k')}, x1, x2, y, sx, sy)
    @addmul! sy * y + sx * x1 * transpose(x2)
end
@inline function binary_einsum!(::SimpleBinaryRule{('i',), ('k',),('k','i')}, x1, x2, y, sx, sy)
    @addmul! sy * y + sx * transpose(x1) * x2
end

# 000
function binary_einsum!(::SimpleBinaryRule{('l',),('l',), ('l',)}, x1, x2, y, sx, sy)
    @addmul! sy * y + sx * x1 * x2
end

# 100
function binary_einsum!(::SimpleBinaryRule{('i','l'),('l',), ('i','l')}, x1, x2, y, sx, sy)
    @addmul! sy * y + sx * x1 * transpose(x2)
end

# 001
@inline function binary_einsum!(::SimpleBinaryRule{('l',), ('k','l'), ('k','l')}, x1, x2, y, sx, sy)
    binary_einsum!(SimpleBinaryRule{('i','l'),('l',),('i','l')}(), x2, x1, y, sx, sy)
end

# 010
function binary_einsum!(::SimpleBinaryRule{('j','l'), ('j','l'), ('l',)}, x1, x2, y, sx, sy)
    @addmul! sy * y + sx * dropdims(mapreduce(*, +, x1, x2; dims=1); dims=1)
end

# 101
function binary_einsum!(::SimpleBinaryRule{('i','l'), ('k','l'), ('i','k','l')}, x1, x2, y::AbstractArray, sx, sy)
    _batched_gemm!('N', 'N', sx, reshape(x1, size(x1, 1), 1, size(x1, 2)), reshape(x2, 1, size(x2, 1), size(x2, 2)), sy, y)
end
@inline function binary_einsum!(::SimpleBinaryRule{('i','l'), ('k','l'), ('k','i','l')}, x1, x2, y::AbstractArray, sx, sy)
    _batched_gemm!('N', 'N', sx, reshape(x2, size(x2, 1), 1, size(x2, 2)), reshape(x1, 1, size(x1, 1), size(x1, 2)), sy, y)
end

# 011
function binary_einsum!(::SimpleBinaryRule{('j','l'), ('j','k','l'), ('k','l')}, x1, x2, y::AbstractArray, sx, sy)
    _batched_gemm!('N', 'N', sx, reshape(x1, 1, size(x1,1), size(x1,2)), x2, sy, reshape(y, 1, size(y,1), size(y,2)))
    y
end
function binary_einsum!(::SimpleBinaryRule{('j','l'), ('k','j','l'), ('k','l')}, x1, x2, y::AbstractArray, sx, sy)
    _batched_gemm!('N', 'T', sx, reshape(x1, 1, size(x1,1), size(x1,2)), x2, sy, reshape(y, 1, size(y,1), size(y,2)))
    y
end

# 110
function binary_einsum!(::SimpleBinaryRule{('i','j','l'), ('j','l'), ('i','l')}, x1, x2, y::AbstractArray, sx, sy)
    _batched_gemm!('N', 'N', sx, x1, reshape(x2, size(x2,1), 1, size(x2,2)), sy, reshape(y, size(y,1), 1, size(y,2)))
    y
end
function binary_einsum!(::SimpleBinaryRule{('j','i','l'), ('j','l'), ('i','l')}, x1, x2, y::AbstractArray, sx, sy)
    _batched_gemm!('T', 'N', sx, x1, reshape(x2, size(x2,1), 1, size(x2,2)), sy, reshape(y, size(y,1), 1, size(y,2)))
    y
end

# ij,jk->ik : 111
# S = N^2
# T = N^3
for (i1, X1) in enumerate([('i', 'j'), ('j', 'i')])
    for (i2, X2) in enumerate([('j', 'k'), ('k', 'j')])
        for (i3, X3) in enumerate([('i', 'k'), ('k', 'i')])
            A1 = i1==i3 ? :(x1) : :(transpose(x1))
            A2 = i2==i3 ? :(x2) : :(transpose(x2))
            @eval function binary_einsum!(::SimpleBinaryRule{$X1,$X2, $X3}, x1, x2, y::AbstractArray{T}, sx, sy) where T
                $(i3==1 ? :(mul!(y, $A1, $A2, sx, sy)) : :(mul!(y, $A2, $A1, sx, sy)))
            end
            X1B = (X1...,'l')
            X2B = (X2...,'l')
            X3B = (X3...,'l')
            C1 = i1==i3 ? 'N' : 'T'
            C2 = i2==i3 ? 'N' : 'T'
            @eval function binary_einsum!(::SimpleBinaryRule{$X1B,$X2B,$X3B}, x1, x2, y::AbstractArray{T}, sx, sy) where T
                $(i3==1 ? :(_batched_gemm!($C1, $C2, sx, x1, x2, sy, y)) : :(_batched_gemm!($C2, $C1, sx, x2, x1, sy, y)))
            end
        end
    end
end