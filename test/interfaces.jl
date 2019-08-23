using Test
using OMEinsum
using OMEinsum: IndexSize, get_size_dict

@testset "index size" begin
    is = IndexSize(('i', 'j'), (2, 3))
    @test is['i'] == 2
    is = is + is
    @test is == IndexSize(('i', 'j', 'i', 'j'), (2, 3, 2, 3))
    @test is['i'] == 2
    a, b, c = randn(2,3), randn(3,4), randn(2,4)
    xs = (a, b, c)
    ixs = (('i', 'j'), ('j', 'k'), ('i', 'k'))
    is = get_size_dict(ixs, xs)
    @test is == IndexSize(('i', 'j', 'j', 'k', 'i', 'k'), (2,3,3,4,2,4))
    @test is['i'] == 2
    @test is['j'] == 3
    @test is['k'] == 4

    @test IndexSize('i'=>2) == IndexSize(('i',), (2,))
end

@testset "unspecified index sizes" begin
    v = randn(5)
    a = randn(6,6)
    xs = (v, a)
    ixs = (('j',), ('k', 'k'))
    @test einsum(ein"j,kk->j", xs, get_size_dict(ixs, xs)) ≈ ein"j,kk->j"(xs...)
    res = ein"j,kk->iij"(xs...; size_info=IndexSize('i'=>9))
    @test size(res) == (9,9,5)
end
