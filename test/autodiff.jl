using Test, OMEinsum, Random
using Zygote
using OMEinsum: tuple_replace, einmagic!

function einsum_bpcheck(ixs, xs, iy, y)
    b = randn(eltype(y), size(y)...)
    @show sum(einmagic!(ixs, xs, iy, y) .* b) |> abs
    #bpcheck(xs -> real(sum(einmagic!(ixs, xs, iy, y))), xs, showy=true)
    bpcheck(xs -> real(sum(einmagic!(ixs, xs, iy, y))), xs, showy=true)
end

@testset "einsum bp" begin
    Random.seed!(3)
    T = ComplexF64
    a = randn(T, 3,3)

    args_list = [
        (((1,2), (2,3)), (a, conj(a)), (1,3), zeros(T,3,3)),
        (((1,2), (2,3)), (a, conj(a)), (1,3), zeros(T,3,3)),
        (((1,2), (1,3),  (1,4)), (a, conj(a), a), (2,3,4), zeros(T,3,3,3)),
        (((1,2), (2,1)), (a, a), (), fill(T(0), ()))
        ]

    for args in args_list
        @test einsum_bpcheck(args...)
    end
end
