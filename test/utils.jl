using OMEinsum: tsetdiff, tunique

@testset "utils" begin
    @test tsetdiff((1,2,3), (2,)) == [1,3]
    @test tunique((1,2,3,3,)) == [1,2,3]
    @test asarray(3) isa Array
    @test asarray(3, randn(3,3)) isa Array
    @test asarray(randn(3,3)) isa Array
end

@testset "allunique" begin
    @test allunique(())
    @test allunique(('i',))
    @test allunique((1,2))
    @test !allunique((1,2,1))
end

@testset "conditioned_permutedims" begin
    a = randn(100, 100)
    OMEinsum.conditioned_permutedims(a, [1,2])
    @test (@allocated OMEinsum.conditioned_permutedims(a, (1,2))) < 100
end

@testset "align_types" begin
    a = randn(100, 100)
    b = randn(ComplexF64, 100, 100)
    res = OMEinsum.align_eltypes(a, b)
    @test res[2] === b
    @test res == (ComplexF64.(a), b)
end
