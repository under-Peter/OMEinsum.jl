using OMEinsum
using StatsBase, Random

function test()
    for i=1:50
        ranka = rand(1:8)
        rankb = rand(1:8)
        ta = [1:ranka...]
        rankab = rand(1:min(ranka, rankb))
        tb = sample(ta, rankab; replace=false)
        tout = setdiff(ta, tb)
        for k=1:rankb-rankab
            push!(tb, ranka+k)
            push!(tout, ranka+k)
        end
        shuffle!(tb)
        shuffle!(tout)
        A = randn(fill(2, ranka)...)
        B = randn(fill(2, rankb)...)
        OMEinsum.batched_contract(Val((ta...,)), A, Val((tb...,)), B, Val((tout...,)))
    end
end

@time test()

function batched_contract2(iAs, A::AbstractArray, iBs, B::AbstractArray, iOuts)
    pA, iAps, iAbs, iAss, pB, iBps, iBbs, iBss, pOut = OMEinsum.analyse_batched_dim(iAs, iBs, iOuts)
    sAb, sAs, sAp, sBs, sBb, sBp, sAB = OMEinsum.analyse_batched_size(iAs, iAps, iAbs, iAss, size(A), iBs, iBps, iBbs, iBss, size(B))

    A, B = OMEinsum.align_eltypes(A, B)
    Apr = reshape(OMEinsum.conditioned_permutedims(A, pA, iAs), sAb, sAs, sAp)
    Bpr = reshape(OMEinsum.conditioned_permutedims(B, pB, iBs), sBs, sBb, sBp)
    AB = OMEinsum._batched_gemm('N','N', Apr, Bpr)
    AB = OMEinsum.conditioned_permutedims(reshape(AB, sAB...), (pOut...,), iOuts)
end

function test2()
    for i=1:10
        ranka = rand(1:8)
        rankb = rand(1:8)
        ta = [1:ranka...]
        rankab = rand(1:min(ranka, rankb))
        tb = sample(ta, rankab; replace=false)
        tout = setdiff(ta, tb)
        for k=1:rankb-rankab
            push!(tb, ranka+k)
            push!(tout, ranka+k)
        end
        shuffle!(tb)
        shuffle!(tout)
        A = randn(fill(2, ranka)...)
        B = randn(fill(2, rankb)...)
        batched_contract2(ta, A, tb, B, tout)
    end
end


@time test2()