using OMEinsum, Test
using OMEinsum: unary_einsum!, Duplicate, Sum, Tr, Permutedims, Repeat, Diag, Identity
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

@testset "Diag" begin
    ix = (3,2,1,1,2)
    iy = (1,2,3)
    size_dict = Dict(1=>3,2=>3,3=>3)
    x = randn(3,3,3,3,3)
    y = randn(3,3,3)
    @test unary_einsum!(Diag(), ix, iy, x, y, true, false) ≈ OMEinsum.loop_einsum(EinCode((ix,),iy), (x,), size_dict)
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

@testset "Identity" begin
    a = rand(5,5,3)
    @test unary_einsum!(Identity(), (1,2,3), (1,2,3), a, ones(5, 5, 3), 2.0, 3.0) ≈ 3 .+ 2a
end

@testset "Sum" begin
    # index-sum
    a = rand(2,2,5)
    @test unary_einsum!(Sum(), (1, 2, 3), (1,2), a, zeros(2, 2), true, false) ≈ sum(a, dims=3)
    a = Basic.(rand(1:100, 2,2,5))
    @test unary_einsum!(Sum(), (1, 2, 3) ,(1,2), a, zeros(Basic, 2, 2), 1, 0) == dropdims(sum(a; dims=3); dims=3)
end