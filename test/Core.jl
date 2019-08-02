using OMEinsum
using OMEinsum: subindex, einindexer

@testset "EinCode" begin
    code = EinCode(((1,2), (2,3)), (1,3))
    @test code isa EinCode
    @test OMEinsum.getixs(code) == ((1,2), (2,3))
    @test OMEinsum.getiy(code) == (1,3)
end

@testset "indexer" begin
    si = einindexer((), ())
    @test subindex(si, (1,2,3)) == 1
    si = einindexer((7,6), (3,2))
    @test OMEinsum.getlocs(si) == (3,2)
    a = randn(7,6)
    @test a[subindex(si, (4,5,2))] == a[2,5]
end
