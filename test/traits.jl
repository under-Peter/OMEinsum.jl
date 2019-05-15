@testset "is_contract, is_decomposible" begin
    @test is_contract(((1,2),(2,3),(1,3)))
    @test !is_contract(((1,2),(1,3), (1,4),(2,3,4)))
    @test !is_decomposible(((1,2),(2,3),(1,3)))
    @test is_decomposible(((1,2),(2,3), (3,4),(1,4)))
end
