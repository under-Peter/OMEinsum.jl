# batched routines
@inline indexpos(iAs, i)::Int = findfirst(==(i), iAs)

# can be used in either static or dynamic invoke
function analyse_batched(iAs, sA, iBs, sB, iOuts)
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

    sAbs = getindex.(Ref(sA), indexpos.(Ref(iAs), iAbs))
    sAb = reduce(*, sAbs, init=1)
    sAs = mapreduce(i->sA[indexpos(iAs,i)], *, iAss, init=1)
    sAps = getindex.(Ref(sA), indexpos.(Ref(iAs),iAps))
    sAp = reduce(*, sAps, init=1)

    sBbs = getindex.(Ref(sB), indexpos.(Ref(iBs),iBbs))
    sBb = reduce(*, sBbs, init=1)
    sBs = mapreduce(i->sB[indexpos(iBs, i)], *, iBss, init=1)
    sBp = mapreduce(i->sB[indexpos(iBs, i)], *, iBps, init=1)
    sAB = vcat(sAbs, sBbs, sAps)
    return pA, sAb, sAs, sAp, pB, sBs, sBb, sBp, sAB, pOut
end

# batched, dynamic version
function batched_contract(iAs, A::AbstractArray, iBs, B::AbstractArray, iOuts::NTuple{NO,T}) where {NO,T}
    pA, sAb, sAs, sAp, pB, sBs, sBb, sBp, sAB, pOut = analyse_batched(iAs, size(A), iBs, size(B), iOuts)

    A, B = align_eltypes(A, B)
    Apr = reshape(conditioned_permutedims(A, pA), sAb, sAs, sAp)
    Bpr = reshape(conditioned_permutedims(B, pB), sBs, sBb, sBp)
    AB = _batched_gemm('N','N', Apr, Bpr)
    AB = conditioned_permutedims(reshape(AB, sAB...), [pOut...])
end

# reload this function for GPU support!
function _batched_gemm(C1::Char, C2::Char, A::StridedArray{T,3}, B::StridedArray{T2,3}) where {T<:Number, T2<:Number}
    batched_gemm(C1, C2, A, B)
end
