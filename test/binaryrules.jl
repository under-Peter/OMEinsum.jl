using OMEinsum, Test

@testset "analyse binary" begin
    size_dict = OMEinsum.IndexSize(1=>1, 2=>2, 3=>3, 4=>4, 6=>6, 7=>7, 8=>8)
    c1, c2, cy, s1, s2, sy, code = OMEinsum.analyze_binary((1,2,3,4,8), (2,6,6,8,4,2), (7,2,1,2,2,6), size_dict)
    @test c1 == (1,4,8,2)
    @test c2 == (4,8,6,2)
    @test cy == (1,6,2)
    @test s1 == (1,32,2)
    @test s2 == (32,6,2)
    @test sy == (1,6,2)
    @test code == ein"ijl,jkl->ikl"
end

@testset "binary rules" begin
    size_dict = OMEinsum.IndexSize(('i', 'j', 'k', 'l'), ntuple(x->5, 4))
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
                    code = EinCode{(i1_,i2_),i3_}()
                    @test einsum(code, (a, b), size_dict) ≈ loop_einsum(code, (a, b), size_dict)
                end
            end
        end
    end
end