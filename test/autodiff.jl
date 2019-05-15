using Test, OMEinsum
using Zygote

@testset "einsum bp" begin
    T = ComplexF64
    a = randn(T, 3,3)

    f1(a) = einsum!(((1,2), (2,3)), (a, conj(a)), (1,3), zeros(T,3,3)) |> sum |> real
    @test bpcheck(f1, a)

    f2(a) = einsum!(((1,2), (1,3),  (1,4)), (a, conj(a), a), (2,3,4), zeros(T,3,3,3)) |> sum |> real
    @test bpcheck(f2, a)

    y = fill(T(0), ())
    @test bpcheck(a->einsum!(((1,2), (2,1)), (a, a), (), y) |> sum |> real, a)
end
