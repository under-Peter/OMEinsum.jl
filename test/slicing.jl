using OMEinsum
using Test, Random

@testset "slice iterator" begin
    se = SlicedEinsum(['i', 'l'], ein"(ij,jk),(kl,lm)->im")
    it = OMEinsum.SliceIterator(se, uniformsize(se, 2))
    @test length(it) == 4
    for (i, v) in enumerate(it)
        @test v == it[i]
    end
end

@testset "SlicedEinsum" begin
    se = SlicedEinsum(['i', 'l'], ein"(ij,jk),(kl,lm)->im")
    @test OMEinsum.flatten(se) == OMEinsum.flatten(se.eins)
    @test labeltype(se) == Char
    xs = (randn(2,3), randn(3,4), randn(4,5), randn(5,6))
    size_info = Dict{Char,Int}()
    @test OMEinsum.get_size_dict!(se, xs, size_info) == Dict('i'=>2, 'j'=>3, 'k'=>4, 'l'=>5, 'm'=>6)
    @test getixsv(se) == [['i','j'],['j','k'],['k','l'],['l','m']]
    @test getiyv(se) == ['i','m']
    @test label_elimination_order(se) == ['j','l', 'k']
    @test se(xs...) ≈ se.eins(xs...)
    @test uniquelabels(se) == ['i', 'j', 'k', 'l', 'm']
    @test uniformsize(se, 2) == Dict(zip(['i', 'j', 'k', 'l', 'm'], ones(Int, 5).*2))
end