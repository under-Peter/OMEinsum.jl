using Test, Combinatorics, Random
using OMEinsum
using OMEinsum: edgesfrominds, operatorfromedge, operatorsfromedges, supportinds,
                indicesafterop, opcost, einsumcost, optimalorder
using OMEinsum: TensorContract, Trace, StarContract, MixedStarContract, Diag,
                MixedDiag, IndexReduction, Permutation, OuterProduct

@testset "einorder" begin
    @testset "edges" begin
        @test edgesfrominds(((1,2),(2,3)), (1,3)) == [2]
        @test edgesfrominds(((1,2),(2,3)), (1,)) == [2,3]
        @test edgesfrominds(((1,2),(2,3)), ()) == [1,2,3]

        @test edgesfrominds(((1,2,3),(2,3,4),(4,5,5,6)), (1,6)) == [2,3,4,5]
        @test edgesfrominds(((1,2,3),(2,3,4),(4,5,5,6)), (1,5,6)) == [2,3,4,5]
    end

    @testset "operator from edge" begin
        @test operatorfromedge(2, ((1,2),(2,3)), (1,3)) isa TensorContract
        @test operatorfromedge(2, ((1,2,2),), (1,)) isa Trace
        @test operatorfromedge(1, ((1,2),(1,3),(1,4)), (2,3,4)) isa StarContract
        @test operatorfromedge(1, ((1,2),(1,3),(1,1)), (2,3,4)) isa MixedStarContract
        @test operatorfromedge(1, ((1,2),(1,3)), (1,2,3)) isa Diag
        @test operatorfromedge(1, ((1,2),(1,3),(1,1)), (1,2,3)) isa MixedDiag
        @test operatorfromedge(3, ((1,2),(1,3),(1,1)), (1,2)) isa IndexReduction
    end

    @testset "supportinds" begin
        @test supportinds(1, ((1,2),(1,3),(1,4))) == (true, true, true)
        @test supportinds(2, ((1,2),(1,3),(1,4))) == (true, false, false)
        @test supportinds(3, ((1,2),(1,3),(1,4))) == (false, true, false)
        @test supportinds(4, ((1,2),(1,3),(1,4))) == (false, false, true)
    end

    @testset "operators from edges" begin
        @test operatorsfromedges(((1,2,3),(2,3,4)), [2,3], (1,4)) == (TensorContract((2,3)),)
        @test operatorsfromedges(((1,2,3),(2,3,4)), [2,3], (4,1)) == (TensorContract((2,3)), Permutation((2,1)))
        @test operatorsfromedges(((1,),(2,),(3,)), [], (1,2,3)) == (OuterProduct{3}(),)
        @test operatorsfromedges(((1,1,2),(2,3)), [1,2], (3,)) == (Trace(1), TensorContract(2))
        @test operatorsfromedges(((1,1,2),(2,3)), [2,1], (3,)) == (TensorContract(2), Trace(1))
        @test operatorsfromedges(((1,1,2),(2,3)), [1,2], (1,3)) == (MixedDiag(1), TensorContract(2))
        @test operatorsfromedges(((1,2),(2,3,4),(1,5,6),(3,5,7,8)), [1,2,3,5], (4,6,7,8)) ==
            (TensorContract(1), TensorContract(2), TensorContract((3,5)), Permutation((2,1,3,4)))
        @test operatorsfromedges(((1,2,3),), [], (1,3,2)) == (Permutation((1,3,2)),)
    end

    @testset "indices after an operation" begin
        op = Trace(1)
        @test indicesafterop(Trace(1), ((1,1,2,3),)) == (2,3)
        @test indicesafterop(TensorContract((1,2)), ((1,2,3),(1,2,4))) == (3,4)
        @test indicesafterop(StarContract(1), ((1,2),(1,3),(1,4))) == (2,3,4)
        @test indicesafterop(Diag(1), ((1,2),(1,3),(1,4))) == (2,3,4,1)
        @test indicesafterop(Permutation((2,3,1)), ('i','j','k')) == ('j','k','i')
        @test indicesafterop(OuterProduct{2}(), ((1,2),(3,4))) == (1,2,3,4)
    end

    @testset "operation cost" begin
        d = 5
        χ = 10
        @test opcost(IndexReduction((1,2)), 0, ((1,2),(3,4)), ((d,d),(d,d)))[1] == d^2
        @test opcost(IndexReduction(1), 0, ((1,2),(3,4)), ((d,d),(d,d)))[1] == d^2
        @test opcost(TensorContract(2), 0, ((1,2),(2,3)), ((d,d),(d,d)))[1] == d^3
        @test opcost(Trace(1), 0, ((1,1),(2,2)), ((d,d), (d,d)))[1] == d
        @test opcost(MixedDiag(1), 0, ((1,1),(2,2)), ((d,d), (d,d)))[1] == d
        @test opcost(StarContract(1), 0, ((1,2),(1,3),(1,4)), ((d,χ),(d,χ),(d,χ)))[1] == d*χ^3
    end

    @testset "einsum cost" begin
        d = 5
        χ = 10
        ops = [IndexReduction('i'), TensorContract('j'), IndexReduction('k')]
        @test einsumcost((('i','j'),('j','k')), ((d,χ),(χ,d)), ops) == d*χ + d*χ + d
        ops = [TensorContract('j'), IndexReduction('i'), IndexReduction('k')]
        @test einsumcost((('i','j'),('j','k')), ((d,χ),(χ,d)), ops) == d^2*χ + d^2 + d
        ops = [TensorContract('j'), IndexReduction(('i','k'))]
        @test einsumcost((('i','j'),('j','k')), ((d,χ),(χ,d)), ops) == χ*d^2 + d^2
        ops = [IndexReduction('i'), IndexReduction('k'), TensorContract('j')]
        @test einsumcost((('i','j'),('j','k')), ((d,χ),(χ,d)), ops) == 2*d*χ + χ

        ops = [Trace('i'), StarContract('k')]
        @test einsumcost((('i','i','k'),('j','k'),('l','k')),
                ((d, d, χ), (d, χ), (d, χ)), ops) == d*χ + χ * d^2
        ops = [StarContract('k'), Trace('i')]
        @test einsumcost((('i','i','k'),('j','k'),('l','k')),
                ((d, d, χ), (d, χ), (d, χ)), ops) == χ*d^3 + d^3
    end

    @testset "optimal contraction" begin
        ixs = ((1,2),(2,3),(3,4))
        xs = (rand(2,3),rand(3,4),rand(4,5))
        optord = optimalorder(ixs, xs, (1,4))
        @test einsumcost(ixs, size.(xs), optord) == 2*3*4 + 2*4*5

        # generate random input, check that random shuffling of the input doesn't
        # change cost.
        ixs = Tuple(Tuple(rand(1:5, rand(1:6))) for i in 1:4)
        sizes = map(x -> map(y -> 5, x), ixs)
        xs = Tuple(randn(s...) for s in sizes)
        cost1 = einsumcost(ixs, sizes, optimalorder(ixs, xs, ()))

        ixs = Tuple(shuffle(collect(ixs)))
        sizes = map(x -> map(y -> 5, x), ixs)
        xs = Tuple(randn(s...) for s in sizes)
        cost2 = einsumcost(ixs, sizes, optimalorder(ixs, xs, ()))
        @test cost1 == cost2

        ixs = Tuple(shuffle(collect(ixs)))
        sizes = map(x -> map(y -> 5, x), ixs)
        xs = Tuple(randn(s...) for s in sizes)
        cost2 = einsumcost(ixs, sizes, optimalorder(ixs, xs, ()))
        @test cost1 == cost2
    end
end
