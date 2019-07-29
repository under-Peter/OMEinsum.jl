using OMEinsum: tsetdiff, tunique

@testset "utils" begin
    @isdefined(pmobj) && next!(pmobj)
    @test tsetdiff((1,2,3), (2,)) == [1,3]
    @isdefined(pmobj) && next!(pmobj)
    @test tunique((1,2,3,3,)) == [1,2,3]
end