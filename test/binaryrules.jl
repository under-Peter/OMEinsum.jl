using OMEinsum, Test
using OMEinsum: SimpleBinaryRule, match_rule
using Polynomials: Polynomial

@testset "analyse binary" begin
    size_dict = Dict(1=>1, 2=>2, 3=>3, 4=>4, 6=>6, 7=>7, 8=>8)
    c1, c2, cy, s1, s2, is, js, ys = OMEinsum.analyze_binary([1,2,3,4,8], [2,6,6,8,4,2], [7,2,1,2,2,6], size_dict)
    @test c1 == [1,4,8,2]
    @test c2 == [4,8,6,2]
    @test cy == [1,6,2]
    @test s1 == [1,32,2]
    @test s2 == [32,6,2]
    @test is == ['i', 'j', 'l']
    @test js == ['j', 'k', 'l']
    @test ys == ['i', 'k', 'l']
end

@testset "binary rules" begin
    size_dict = Dict(zip(('i', 'j', 'k', 'l'), ntuple(x->5, 4)))
    nmatch = 0
    for has_batch in [true, false]
        for i1 in [(), ('i',), ('j',), ('i','j'), ('j', 'i')]
            for i2 in [(), ('k',), ('j',), ('k','j'), ('j', 'k')]
                for i3 in [(), ('i',), ('k',), ('i','k'), ('k', 'i')]
                    @info i1, i2, i3, has_batch
                    i1_ = has_batch ? (i1..., 'l') : i1
                    i2_ = has_batch ? (i2..., 'l') : i2
                    i3_ = has_batch ? (i3..., 'l') : i3
                    a = randn(fill(5, length(i1_))...) |> OMEinsum.asarray
                    b = randn(fill(5, length(i2_))...) |> OMEinsum.asarray
                    code = EinCode((i1_,i2_),i3_)
                    rule = match_rule(code)
                    if rule isa SimpleBinaryRule
                        nmatch += 1
                        @test einsum(rule, (a, b)) ≈ loop_einsum(code, (a, b), size_dict)
                    else
                        @test einsum(code, (a, b), size_dict) ≈ loop_einsum(code, (a, b), size_dict)
                    end
                end
            end
        end
    end
    @test nmatch == 36
end

@testset "match binary rules" begin
    for code in [ein",->", ein",k->k", ein"i,->i", ein"j,j->", ein"i,k->ik", ein"i,k->ki",
                ein"j,jk->k", ein"j,kj->k", ein"ji,j->i", ein"ij,j->i", ein"ji,jk->ik",
                ein"ji,kj->ik", ein"ji,jk->ki", ein"ji,kj->ki", ein"ij,jk->ik", ein"ij,kj->ik", ein"ij,jk->ki", ein"ij,kj->ki"]
        @test OMEinsum.match_rule(code) == SimpleBinaryRule(code)
    end
    OMEinsum.match_rule(ein"ab,bc->ac") == SimpleBinaryRule(('i', 'j'), ('j', 'k'), ('i', 'k'))
    OMEinsum.match_rule(ein"ab,bce->ac") == OMEinsum.DefaultRule()
end 

@testset "binary einsum" begin
    for code in [
        ein",->",
        ein"ijl,jl->il",
        ein"ab,bc->ac",
        ein"abb,bc->ac",  # with diag in
        ein"ab,bc->acc",  # with diag out
        ein"ab,bce->ac",  # with sum in
        ein"ab,bc->ace",  # with sum out
        ein"bal,bcl->lcae",  # with perm in
        ein"bal,bcl->ca",  # with multi-edge in
        ein"bal,bc->lca",  # with multi-edge out
        ein"ddebal,bcf->lcac",  # with all
    ]
        xs = [OMEinsum.asarray(randn(ComplexF64, fill(5, length(ix))...)) for ix in OMEinsum.getixs(code)]
        size_dict = Dict(zip(('a', 'b', 'c', 'd', 'e', 'f','i','j','k','l'), ntuple(x->5, 10)))
        @test einsum(code, (xs...,), size_dict) ≈ loop_einsum(code, (xs...,), size_dict)
    end
end

@testset "regression test" begin
    x, y = randn(fill(2, 8)...), randn(fill(2, 5)...)
    code = EinCode(((10, 5, 4, 2, 6, 6, 7, 9), (1, 9, 8, 2, 10)), (4, 10, 9, 7, 10))
    @test code(x, y) ≈ loop_einsum(code, (x, y), OMEinsum.get_size_dict(OMEinsum.getixs(code), (x, y)))
end

@testset "polynomial scalar mul" begin
    @test ein",->"(OMEinsum.asarray(Polynomial([1.0])), OMEinsum.asarray(Polynomial([1.0]))) isa AbstractArray
end
