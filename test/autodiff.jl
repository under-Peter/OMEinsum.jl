using Test
using OMEinsum: bpcheck, einsum!
using OMEinsum
using Zygote

@testset "einsum! bp" begin
    for T in (Float64, ComplexF64)
        @testset "$T" begin
            # matrix and vector multiplication
            a,b,c = rand(T,2,2), rand(T,2,2), rand(T,2,2)
            v = rand(T,2)
            t = randn(2,2,2,2)
            @test bpcheck( (a,b,c) -> einsum!(((1,2),(2,3),(3,4)), (a,b,c), (1,4), zeros(T,2,2)) |> abs ∘ sum ,a,b,c)

            @test bpcheck( (a,b,c) -> einsum!(((1,2),(2,3),(3,4)), (a,b,c), (4,1), zeros(T,2,2)) |> abs ∘ sum ,a,b,c)

            @test bpcheck((a,v) -> einsum!(((1,2),(2,)), (a,v), (1,), zeros(T,2)) |> abs ∘ sum , a, v)

            # contract to 0-dim array
            @test bpcheck((a,b) -> einsum!(((1,2),(1,2)), (a,b), (), zeros(T)) |> abs ∘ sum , a,b)

            # trace
            @test bpcheck(a -> einsum!(((1,1),), (a,), (), zeros(T)) |> abs ∘ sum, a)
            aa = rand(T,2,4,4,2)
            @test bpcheck(aa -> einsum!(((1,2,2,1),), (aa,), (), zeros(T)) |> abs ∘ sum, aa)


            # partial trace
            @test bpcheck(aa -> einsum!(((1,2,2,3),), (aa,), (1,3), zeros(T,2,2)) |> abs ∘ sum, aa)

            # diag
            @test bpcheck(aa -> einsum!(((1,2,2,3),), (aa,), (1,2,3), zeros(T,2,4,2)) |> abs ∘ sum, aa)

            # permutation
            @test bpcheck(a -> einsum!(((1,2),), (a,), (2,1), zeros(T,2,2)) |> abs ∘ sum, a)
            @test bpcheck(t -> einsum!(((1,2,3,4),), (t,),(2,3,1,4), zeros(T,2,2,2,2)) |> abs ∘ sum, t)

            # tensor contraction
            @test bpcheck((t,a) -> einsum!(((1,2,3,4), (2,3)), (t,a), (1,4), zeros(T,2,2)) |> abs ∘ sum, t,a)
            @test bpcheck((t,a) -> einsum!(((4,3,2,1), (2,3)), (t,a), (1,4), zeros(T,2,2)) |> abs ∘ sum, t,a)

            # star-contraction
            @test bpcheck((a,b,c) -> einsum!(((1,2),(1,3),(1,4)), (a,b,c), (2,3,4), zeros(T,2,2,2)) |> abs ∘ sum, a,b,c)

            # star and contract
            @test bpcheck((a,b,c) -> einsum!(((1,2),(1,2),(1,3)), (a,b,c), (3,), zeros(T,2)) |> abs ∘ sum, a,b,c)

            # index-sum
            a3 = rand(T,2,2,2)
            @test bpcheck(a -> einsum!(((1,2,3),),(a,),(1,2), zeros(T,2,2)) |> abs ∘ sum, a3)

            # Hadamard product
            @test bpcheck((a,b) -> einsum!(((1,2),(1,2)), (a,b), (1,2), zeros(T,2,2)) |> abs ∘ sum, a, b)

            # Outer
            @test bpcheck((a,b) -> einsum!(((1,2),(3,4)),(a,b),(1,2,3,4), zeros(T,2,2,2,2)) |> abs ∘ sum, a, b)
        end
    end
end

@testset "einsum bp" begin
    for T in (Float64, ComplexF64)
        @testset "$T" begin
            # matrix and vector multiplication
            a,b,c = rand(T,2,2), rand(T,2,2), rand(T,2,2)
            v = rand(T,2)
            t = randn(2,2,2,2)
            @test bpcheck( (a,b,c) -> einsum(((1,2),(2,3),(3,4)), (a,b,c), (1,4)) |> abs ∘ sum ,a,b,c)
            @test bpcheck( (a,b,c) -> einsum(((1,2),(2,3),(3,4)), (a,b,c), (4,1)) |> abs ∘ sum ,a,b,c)
            @test bpcheck((a,v) -> einsum(((1,2),(2,)), (a,v), (1,)) |> abs ∘ sum , a, v)

            # contract to 0-dim array
            @test bpcheck((a,b) -> einsum(((1,2),(1,2)), (a,b), ()) |> abs ∘ sum , a,b)

            # trace
            @test bpcheck(a -> einsum(((1,1),), (a,), ()) |> abs ∘ sum, a)
            aa = rand(T,2,4,4,2)
            @test bpcheck(aa -> einsum(((1,2,2,1),), (aa,), ()) |> abs ∘ sum, aa)


            # partial trace
            @test bpcheck(aa -> einsum(((1,2,2,3),), (aa,), (1,3)) |> abs ∘ sum, aa)

            # diag
            @test bpcheck(aa -> einsum(((1,2,2,3),), (aa,), (1,2,3)) |> abs ∘ sum, aa)

            # permutation
            @test bpcheck(a -> einsum(((1,2),), (a,), (2,1)) |> abs ∘ sum, a)
            @test bpcheck(t -> einsum(((1,2,3,4),), (t,),(2,3,1,4)) |> abs ∘ sum, t)

            # tensor contraction
            @test bpcheck((t,a) -> einsum(((1,2,3,4), (2,3)), (t,a), (1,4)) |> abs ∘ sum, t,a)
            @test bpcheck((t,a) -> einsum(((4,3,2,1), (2,3)), (t,a), (1,4)) |> abs ∘ sum, t,a)

            # star-contraction
            @test bpcheck((a,b,c) -> einsum(((1,2),(1,3),(1,4)), (a,b,c), (2,3,4)) |> abs ∘ sum, a,b,c)

            # star and contract
            @test bpcheck((a,b,c) -> einsum(((1,2),(1,2),(1,3)), (a,b,c), (3,)) |> abs ∘ sum, a,b,c)

            # index-sum
            a3 = rand(T,2,2,2)
            @test bpcheck(a -> einsum(((1,2,3),),(a,),(1,2)) |> abs ∘ sum, a3)

            # Hadamard product
            @test bpcheck((a,b) -> einsum(((1,2),(1,2)), (a,b), (1,2)) |> abs ∘ sum, a, b)

            # Outer
            @test bpcheck((a,b) -> einsum(((1,2),(3,4)),(a,b),(1,2,3,4)) |> abs ∘ sum, a, b)
        end
    end
end

@testset "gradient type check" begin
    array_match(x, y) = typeof(x) == typeof(y) && size(x) == size(y)
    a = randn(3,3)
    b = randn(3,3)
    @test array_match(gradient(a->einsum(((1,2), (2,1)), (a, b), ())[] |> abs, a)[1], a)
    b = randn(ComplexF64,3,3)
    @test_broken array_match(gradient(a->einsum(((1,2), (2,1)), (a, b), ())[] |> abs, a)[1], a)
    a = randn(ComplexF64,3,3)
    @test array_match(gradient(a->einsum(((1,2), (2,3)), (a, b), ())[] |> abs, a)[1], a)
    b = randn(3,3)
    @test array_match(gradient(a->einsum(((1,2), (2,3)), (a, b), ())[] |> abs, a)[1], a)
end
