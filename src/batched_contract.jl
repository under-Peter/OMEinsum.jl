# batched routines
@inline indexpos(iAs, i)::Int = findfirst(==(i), iAs)

# can be used in either static or dynamic invoke
function analyse_batched_dim(iAs, iBs, iOuts)
    iABs = iAs ∩ iBs
    pres   = iABs ∩ iOuts
    broad  = setdiff((iAs ∩ iOuts) ∪ (iBs ∩ iOuts), pres)
    summed = setdiff(iABs, pres)

    iAps, iAbs, iAss = pres ∩ iAs, broad ∩ iAs, summed ∩ iAs
    iBps, iBbs, iBss = pres ∩ iBs, broad ∩ iBs, summed ∩ iBs

    pA   = indexpos.(Ref(iAs), vcat(iAbs, iAss, iAps))
    pB   = indexpos.(Ref(iBs), vcat(iBss, iBbs, iBps))
    iABs = vcat(iAbs, iBbs, iAps)
    pOut = indexpos.(Ref(iABs), iOuts)
    return (pA...,), (iAps...,), (iAbs...,), (iAss...,), (pB...,), (iBps...,), (iBbs...,), (iBss...,), pOut
end

function analyse_batched_size(iAs, iAps, iAbs, iAss, sA, iBs, iBps, iBbs, iBss, sB)
    sAbs = getindex.(Ref(sA), indexpos.(Ref(iAs), iAbs))
    sAb = reduce(*, sAbs, init=1)
    sAs = mapreduce(i->sA[indexpos(iAs,i)], *, iAss, init=1)
    sAps = getindex.(Ref(sA), indexpos.(Ref(iAs),iAps))
    sAp = reduce(*, sAps, init=1)

    sBbs = getindex.(Ref(sB), indexpos.(Ref(iBs),iBbs))
    sBb = reduce(*, sBbs, init=1)
    sBs = mapreduce(i->sB[indexpos(iBs, i)], *, iBss, init=1)
    sBp = mapreduce(i->sB[indexpos(iBs, i)], *, iBps, init=1)
    sAB = (sAbs..., sBbs..., sAps...)
    return sAb, sAs, sAp, sBs, sBb, sBp, sAB
end

# batched, dynamic version
@generated function batched_contract(::Val{iAs}, A::AbstractArray, ::Val{iBs}, B::AbstractArray, ::Val{iOuts}) where {iAs, iBs, iOuts, NO,T}
    pA, iAps, iAbs, iAss, pB, iBps, iBbs, iBss, pOut = analyse_batched_dim(iAs, iBs, iOuts)
    exA = any(i-> (@inbounds pA[i]!=i), 1:length(pA)) ? :(tensorpermute(A, $pA)) : :(A)
    exB = any(i-> (@inbounds pB[i]!=i), 1:length(pB)) ? :(tensorpermute(B, $pB)) : :(B)
    exAB = any(i-> (@inbounds pOut[i]!=i), 1:length(pOut)) ? :(tensorpermute(AB, $pOut)) : :(AB)
    quote
        sAb, sAs, sAp, sBs, sBb, sBp, sAB = analyse_batched_size($iAs, $iAps, $iAbs, $iAss, size(A), $iBs, $iBps, $iBbs, $iBss, size(B))

        A, B = align_eltypes(A, B)
        Apr = reshape($exA, sAb, sAs, sAp)
        Bpr = reshape($exB, sBs, sBb, sBp)
        AB = reshape(_batched_gemm('N','N', Apr, Bpr), sAB...)
        AB = $exAB
    end
end

# reload this function for GPU support!
function _batched_gemm(C1::Char, C2::Char, A::StridedArray{T,3}, B::StridedArray{T2,3}) where {T<:BlasFloat, T2<:BlasFloat}
    batched_gemm(C1, C2, A, B)
end

function _batched_gemm(C1::Char, C2::Char, A::AbstractArray{T,3}, B::AbstractArray{T2,3}) where {T<:BlasFloat, T2<:BlasFloat}
    batched_gemm(C1, C2, Array(A), Array(B))
end
