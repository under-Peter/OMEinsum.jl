using Test
using TupleTools
using OMEinsum: dedup, redup, Trace, Diag, TensorContract

@testset "einevaluate" begin
    @testset "index de- and reduplication" begin
        op = Diag('i')
        ixs = (('i','j'),('i','j'))
        nixs, rev = dedup(ixs, op)
        # no duplicates except operated on
        allnixs = TupleTools.vcat(nixs...)
        @test all(i -> i ∈ op.edges || count(==(i), allnixs) == 1, allnixs)
        @test all(i -> ixs[i] == redup(nixs[i], rev), 1:length(ixs))

        op = Trace(('i','k'))
        ixs = (('i','k','j','j','k','i'),)
        nixs, rev = dedup(ixs, op)
        # no duplicates except operated on
        allnixs = TupleTools.vcat(nixs...)
        @test all(i -> i ∈ op.edges || count(==(i), allnixs) == 1, allnixs)
        @test all(i -> ixs[i] == redup(nixs[i], rev), 1:length(ixs))

        op = TensorContract(('i','k'))
        ixs = (('l','l','i','k'),('i','k','m','m'))
        nixs, rev = dedup(ixs, op)
        # no duplicates except operated on
        allnixs = TupleTools.vcat(nixs...)
        @test all(i -> i ∈ op.edges || count(==(i), allnixs) == 1, allnixs)
        @test all(i -> ixs[i] == redup(nixs[i], rev), 1:length(ixs))
    end
end
