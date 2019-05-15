module OMEinsum
using BatchedRoutines

export pairwise_contract, einsum


indexpos(iAs, i) = findfirst(==(i), iAs)

@doc raw"
   greet()
say hello to the world!
"
greet() = print("Hello World!")
function pairwise_contract(iAs, A, iBs, B, iOuts)
    iABs = iAs ∩ iBs
    pres   = iABs ∩ iOuts
    broad  = setdiff((iAs ∩ iOuts) ∪ (iBs ∩ iOuts), pres)
    summed = setdiff(iABs, pres)

    iAps, iAbs, iAss = pres ∩ iAs, broad ∩ iAs, summed ∩ iAs
    iBps, iBbs, iBss = pres ∩ iBs, broad ∩ iBs, summed ∩ iBs

    pA   = indexpos.(Ref(iAs), vcat(iAbs, iAss, iAps))
    pB   = indexpos.(Ref(iBs), vcat(iBss, iBbs, iBps))
    iABs = vcat(iAbs, iBbs, iAps)
    pOut = indexpos.(Ref(iOuts), iABs)

    sA, sB = size(A), size(B)
    sAbs = Int[sA[i] for i in indexpos.(Ref(iAs), iAbs)]
    sAb = prod(sAbs)
    sAs = prod(Int[sA[i] for i in indexpos.(Ref(iAs),iAss)])
    sAps = Int[sA[i] for i in indexpos.(Ref(iAs),iAps)]
    sAp = prod(sAps)
    sBbs = Int[sB[i] for i in indexpos.(Ref(iBs),iBbs)]
    sBb = prod(sBbs)
    sBs = prod(Int[sB[i] for i in indexpos.(Ref(iBs),iBss)])
    sBp = prod(Int[sB[i] for i in indexpos.(Ref(iBs),iBps)])
    sAB = vcat(sAbs, sBbs, sAps)

    Apr = reshape(permutedims(A, pA), sAb, sAs, sAp)
    Bpr = reshape(permutedims(B, pB), sBs, sBb, sBp)
    AB = batched_gemm('N','N', Apr, Bpr)
    AB = permutedims(reshape(AB, sAB...), invperm(pOut))
end

function einsum(contractions, tensors, outinds)
    if length(contractions) == 2
        return pairwise_contract(contractions[1], tensors[1],
                          contractions[2], tensors[2],
                          outinds)
    end

    c1, c2 = contractions[1], contractions[2]
    cs = reduce(∪, collect.(contractions[3:end]))
    broad = symdiff(c1,c2)
    summed = c1 ∩ c2 ∩ (cs ∪ outinds)
    out = collect(broad ∪ summed)
    t1, t2 = tensors[1], tensors[2]
    t1t2 = pairwise_contract(c1, t1, c2, t2, out)
    return einsum((out, contractions[3:end]...), (t1t2, tensors[3:end]...), outinds)
end

end # module
