using OMEinsum, Test
using OMEinsum: unary_einsum!, Duplicate, Sum, Tr, Permutedims, Repeat
using SymEngine: Basic

@testset "Duplicate" begin
    ix = (1,2,3)
    iy = (3,2,1,1,2)
    size_dict = Dict(1=>3,2=>3,3=>3)
    x = randn(3,3,3)
    y = randn(3,3,3,3,3)
    @test OMEinsum.duplicate!(y, x, ix, iy, true, false) ≈ OMEinsum.loop_einsum(EinCode((ix,),iy), (x,), size_dict)
    @test unary_einsum!(Duplicate(), ix, iy, x, y, true, false) ≈ OMEinsum.loop_einsum(EinCode((ix,),iy), (x,), size_dict)
end

@testset "Repeat" begin
    ix = (1,2,3)
    iy = (3,4,2,1)
    size_dict = Dict(1=>3,2=>3,3=>3,4=>5)
    x = randn(3,3,3)
    y = randn(3,5,3,3)
    @test unary_einsum!(Repeat(), ix, iy, x, y, true, false) ≈ OMEinsum.loop_einsum(EinCode((ix,),iy), (x,), size_dict)
end

@testset "Tr" begin
    a = rand(5,5)
    @test unary_einsum!(Tr(), (1,1),(), a, fill(1.0), true, false)[] ≈ sum(a[i,i] for i in 1:5)
    a = Basic.(rand(5,5))
    @test isapprox(unary_einsum!(Tr(), (1,1),(), a, fill(Basic(0)), 1, 0)[], sum(a[i,i] for i in 1:5), rtol=1e-8)
end

@testset "Permutedims" begin
    a = rand(5,5,3)
    @test unary_einsum!(Permutedims(), (1,2,3), (2,3,1), a, zeros(5, 3, 5), true, false) ≈ permutedims(a, (2,3,1))
end

@testset "Sum" begin
    # index-sum
    a = rand(2,2,5)
    @test unary_einsum!(Sum(), (1, 2, 3), (1,2), a, zeros(2, 2), true, false) ≈ sum(a, dims=3)
    a = Basic.(rand(1:100, 2,2,5))
    @test unary_einsum!(Sum(), (1, 2, 3) ,(1,2), a, zeros(Basic, 2, 2), 1, 0) == dropdims(sum(a; dims=3); dims=3)
end

@testset "allow loops" begin
    t = rand(5,5,5,5)
    a = rand(5,5)
    size_dict = Dict(zip((1,2,3,4,2,3), ((size(t)..., size(a)...))))

    OMEinsum.allow_loops(false)
    @test_throws ErrorException loop_einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict)
    OMEinsum.allow_loops(true)

    ta = loop_einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict)
    @test einsum!(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict) ≈  ta
    @test einsum!(DefaultRule(), ((1,2,3,4), (2,3)), (1,4), (t,a), size_dict) ≈  ta

    # index-sum
    t = Basic.(rand(5,5,5,5))
    a = Basic.(rand(5,5))
    size_dict = Dict(zip((1,2,3,4,2,3), ((size(t)..., size(a)...))))
    ta = loop_einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict)
    @test einsum!(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict) ≈  ta
    @test unary_einsum!(DefaultRule(), ((1,2,3,4), (2,3)), (1,4), (t,a), size_dict) ≈  ta
end

