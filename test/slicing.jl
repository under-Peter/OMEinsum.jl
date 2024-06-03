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
    expected = se.eins(xs...)
    @test se(xs...) ≈ expected
    y = similar(se(xs...))
    @test einsum!(se, xs, y, true, false, size_info) ≈ expected
    @test y ≈ expected
    @test uniquelabels(se) == ['i', 'j', 'k', 'l', 'm']
    @test uniformsize(se, 2) == Dict(zip(['i', 'j', 'k', 'l', 'm'], ones(Int, 5).*2))
end

@testset "replace" begin
    code = ein"(ij, jk), kl->il"
    se = optimize_code(code, uniformsize(code, 2), TreeSA(;niters=0))
    se2 = replace(se, 'i'=>'a', 'j'=>'b', 'k'=>'c', 'l'=>'d')
    @test labeltype(se2) == Char
    @test se2 == SlicedEinsum(Char[], ein"(ab, bc), cd->ad")
end