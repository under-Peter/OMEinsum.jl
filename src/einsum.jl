@doc raw"
    einsum(::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}

return the tensor that results from contracting the tensors `xs` according
to their indices `ixs`, where all indices that do not appear in the output `iy` are
summed over.
The result is permuted according to `out`.

- `ixs` - tuple of tuples of index-labels of the input-tensors `xs`

- `iy` - tuple of index-labels of the output-tensor

- `xs` - tuple of tensors

- `size_dict` - `IndexSize`-object that maps index-labels to their sizes

# example

```jldoctest; setup = :(using OMEinsum)
julia> a, b = rand(2,2), rand(2,2);

julia> einsum(EinCode((('i','j'),('j','k')),('i','k')), (a, b)) ≈ a * b
true

julia> einsum(EinCode((('i','j'),('j','k')),('k','i')), (a, b)) ≈ permutedims(a * b, (2,1))
true
```
"
@generated function einsum(code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    rule = match_rule(ixs, iy)
    :(einsum($rule, code, xs, size_dict))
end

function einsum(::Tr, ::EinCode, xs, size_dict)
    @debug "Tr" size.(xs)
    asarray(tr(xs[1]), xs[1])
end

using TensorOperations

# note that dispatching to Hadamard if some `ixs` are permuted has inferior
# performance compared to the fallback
function einsum(::Hadamard, ::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    @debug "Hadamard" ixs => iy size.(xs)
    asarray(broadcast(*, xs...), xs[1])
end

function einsum(::PairWise, ::EinCode{ixs, iy}, xs::NTuple{NT,AbstractArray}, size_dict) where {ixs, iy, NT}
    @debug "PairWise optcontract" ixs => iy size.(xs)
    optcontract(ixs, xs, iy)
end

function einsum(sm::Sum, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    dims = (findall(i -> i ∉ iy, ixs[1])...,)
    (ix1,) = ixs
    ix1f = filter!(i -> i in iy, collect(ix1))
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

@generated function einsum(sm::MatMul, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    ix1, ix2 = ixs
    l = ifelse(ix1[1] in ix2, ix1[1], ix1[2])
    if ix1[2] == l && ix2[1] == l
        if iy == (ix1[1], ix2[2])
            return :(@debug "MatMul1 (ij,jk -> ik)" ixs => iy size.(xs);
                xs[1] * xs[2]
            )
        else
            return :(@debug "MatMul2 (ij,jk -> ki) permutedims(ik)" ixs => iy size.(xs);
                permutedims(xs[1] * xs[2])
            )
        end
    elseif ix1[1] == l && ix2[1] == l
        if iy == (ix1[2], ix2[2])
            return :(@debug "MatMul3 (ji,jk -> ik) transpose(ji)" ixs => iy size.(xs);
                transpose(xs[1]) * xs[2]
            )
        else
            return :(@debug "MatMul4 (ji,jk -> ki) transpose(jk)" ixs => iy size.(xs);
                transpose(xs[2]) * xs[1]
            )
        end
    elseif ix1[2] == l && ix2[2] == l
        if iy == (ix1[1], ix2[1])
            return :(@debug "MatMul5 (ij,kj -> ik) transpose(kj)" ixs => iy size.(xs);
                xs[1] * transpose(xs[2])
            )
        else
            return :(@debug "MatMul6 (ij,kj -> ki) transpose(ij)" ixs => iy size.(xs);
                xs[2] * transpose(xs[1])
            )
        end
    else #ix1[1] == l && ix2[2] == l
        if iy == (ix1[2], ix2[1])
            return :(@debug "MatMul7 (ji,kj -> ik) permutedims(ki)" ixs => iy size.(xs);
                permutedims(xs[2] * xs[1])
            )
        else
            return :(@debug "MatMul8 (ji,kj -> ki)" ixs => iy size.(xs);
                xs[2] * xs[1]
            )
        end
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

function einsum(::BatchedContract, code::EinCode{ixs, iy}, xs::NTuple{NT, Any}, size_dict) where {ixs, iy, NT}
    @debug "BatchedContract loop_einsum" ixs => iy size.(xs)
    loop_einsum(code, xs, size_dict)
end

@generated function _preprocess_dupindices(::Val{ix}, x) where {ix}
    if length(tunique(ix)) != length(ix)
        iy = [l for l in ix if count(==(l), ix) == 1]
        :(($(Val((iy...,))), einsum($(EinCode((ix,), (iy...,))), (x,), get_size_dict(($ix,), (x,)))))
    else
        :(($(Val(ix)), x))
    end
end

@generated function einsum(::BatchedContract, ::EinCode{ixs,iy}, xs::NTuple{<:Any, AbstractArray{<:BlasFloat}}, size_dict) where {ixs, iy}
    quote
        ixs1, xs1 = _preprocess_dupindices($(Val(ixs[1])), xs[1])
        ixs2, xs2 = _preprocess_dupindices($(Val(ixs[2])), xs[2])
        @debug "BatchedContract" ixs => iy ixs1 ixs2 size(xs1) size(xs2)
        batched_contract(ixs1, xs1, ixs2, xs2, $(Val(iy)))
    end
end
