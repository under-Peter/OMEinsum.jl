using Test
using OMEinsum
using OMEinsum: get_size_dict

@testset "unspecified index sizes" begin
    v = randn(5)
    a = randn(6,6)
    xs = (v, a)
    ixs = (('j',), ('k', 'k'))
    @test einsum(ein"j,kk->j", xs, get_size_dict(ixs, xs)) ≈ ein"j,kk->j"(xs...)
    res = ein"j,kk->iij"(xs...; size_info=Dict('i'=>9))
    @test size(res) == (9,9,5)
    @test ein"ijk,
    ijk->
    ikl" == ein"ijk,ijk->ikl"
end
