using OMEinsum: _unique
using OMEinsum
using Test

@testset "utils" begin
    @test _unique(Int,(1,2,3,3,)) == [1,2,3]
    @test asarray(3) isa Array
    @test asarray(3, randn(3,3)) isa Array
    @test asarray(randn(3,3)) isa Array
    @test asscalar(fill(3,())) === 3
    @test asscalar(3) === 3
end

@testset "allunique" begin
    @test allunique(())
    @test allunique(('i',))
    @test allunique((1,2))
    @test !allunique((1,2,1))
end

@testset "tensorpermute" begin
    a = randn(100, 100)
    @test OMEinsum.tensorpermute(a, [1,2]) == a
    @test OMEinsum.tensorpermute(a, (2,1)) == transpose(a)
end

@testset "align_types" begin
    a = randn(100, 100)
    b = randn(ComplexF64, 100, 100)
    res = OMEinsum.align_eltypes(a, b)
    @test res[2] === b
    @test res == (ComplexF64.(a), b)
end

@testset "batched gemm" begin
    A = randn(10, 10, 10)
    B = randn(10, 10, 10)
    for C1 in ['N', 'T']
        for C2 in ['N', 'T']
            A_ = Array{Any}(A)
            B_ = Array{Any}(B)
            @test OMEinsum._batched_gemm(C1, C2, A, B) ≈ OMEinsum._batched_gemm(C1, C2, A_, B_)
        end
    end
end

@testset "addmul!" begin
    x = randn(10, 10)
    y = randn(10, 10)
    z = randn(10, 10)
    for a in [0.0, 1.0, 4.0], b in [0.0, 1.0, 4.0]
        @test (o = copy(x); OMEinsum.@addmul! a * o + b * y * z) ≈ a .* x .+ b .* y .* z
    end
end