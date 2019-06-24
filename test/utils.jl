@testset "utils" begin
    @test setdiff((1,2,3), (2,)) == [1,3]
    @test unique((1,2,3,3,)) == [1,2,3]
end
