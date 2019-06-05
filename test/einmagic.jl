using Test, OMEinsum, SimpleTraits
using OMEinsum: einmagic!, einsum!
using BenchmarkTools

@testset "eincode" begin
    @test_broken EinCode(((1,2), (2,3)), (1,3)) isa EinCode{((1,2), (2,3), (1,3))}
end

@testset "trait function" begin
    # input is (ixs..., iy)
    @test_broken is_contract(((1,2),(2,3),(1,3)))
    @test_broken !is_contract(((1,2),(1,3), (1,4),(2,3,4)))
    @test_broken is_trace(((2,2),()))
    @test_broken !is_trace(((2,),()))
    @test_broken is_sum(((1,2,3,6),(1,2)))
    @test_broken !is_sum(((1,2,3,3),()))
    @test_broken is_dot(((1,),(1,),()))
    @test_broken !is_dot(((1,),(3,),()))
    @test_broken is_permute(((1,2,3,6),(6,2,3,1)))
    @test_broken !is_permute(((1,2,3,6),(6,2,3,2)))
    @test_broken is_batched_matmul(((1,2,3),(2,3,6),(6,1,3)))
    @test_broken !is_batched_matmul(((1,2,3),(2,7,6),(6,1,3)))

    # not supported traits (probably no speedup)
    # * hadamard
    # * ptrace
    # * diag
    # * indexsum
    # * outer
    # * star
end

@testset "trait dispatching" begin
    asarray(x::Number) = fill(x, ())

    acc = 1.2  # the acceleration ratio
    # for dispatching
    N = 100
    a = randn(N, N)
    t = randn(N, N, N)
    v = randn(N)
    @test_broken istrait(IsPairWise{((1,2), (2,3), (1,3))})
    args_list = [
        (((1,2), (2,3)), (a,a), (1,3), randn(N,N)),  # contract
        (((1,1),), (a,), (), asarray(0.0)),   # trace
        (((1,2), (1,3), (1,4)), (a,a,a), (2,3,4), randn(N,N,N)),  # star
        (((1,2,),), (a,), (1,), randn(N)),   # sum
        (((1,),(1,)), (v,v), (), asarray(0.0)),  # dot
        (((1,2,3),(1,3,4)), (t,t), (2,4,3), randn(N,N,N)),  # batched_matmul
        (((1,2,3),), (t,), (2,1,3), randn(N,N,N))  # permute
    ]
    for args in args_list
        @test_broken einmagic!(EinCode{Tuple(args[1]..., args[3])}, args[2], args[4]) ≈ einsum!(args...)
        @test_broken einmagic!(args...) ≈ einsum!(args...)
        @test_broken acc * BenchmarkTools.median((@benchmark einmagic!(args...)).times) < BenchmarkTools.median((@benchmark einsum!(args...)).times)
    end
end
