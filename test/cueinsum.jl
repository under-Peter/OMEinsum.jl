using Test
using OMEinsum
using CuArrays
using DoubleFloats

CuArrays.allowscalar(false)

@testset "cuda einsum" begin
    a = [randn(fill(3, i)...) for i=1:4]
    ca = a .|> CuArray
    ixs = ((1,2), (2,3))
    xs = (ca[2], ca[2])
    @test loop_einsum!(EinCode(ixs, (1,3)), xs, zeros(3,3) |> CuArray, OMEinsum.get_size_dict(ixs, xs)) ≈ ca[2]*ca[2]
    for f in [ein"ij,jk->ik", ein"ii->", ein"ijj ->i", ein"ij,ik,il->jkl", ein"ii->i", ein"ijl->i", ein"i->ii", ein"ij,jk,kl->il", ein"ij,ij,ij -> ij"]
        @show f
        cins = map(ix->ca[length(ix)], OMEinsum.getixs(f))
        ins = map(ix->a[length(ix)], OMEinsum.getixs(f))
        @test f(cins...) isa CuArray
        @test f(ins...) ≈ f(cins...)
    end
end

@testset "fallback - getindex IR error" begin
    a = rand(ComplexF64,2,2,2)
    ca = CuArray(a);
    @test Array(ein"npu,por,dom,lmn -> urdl"(ca,ca,ca,ca)) ≈ ein"npu,por,dom,lmn -> urdl"(a,a,a,a)
end

@testset "isbatchmul" begin
    for (ixs, iy) in [(((1,2), (2,3)), (1,3)), (((1,2,3), (2,3)), (1,3)),
                        (((7,1,2,3), (2,4,3,7)), (1,4,3)),
                        (((3,), (3,)), (3,)), (((3,1), (3,)), (3,1))
                        ]
        xs = ([randn(ComplexF64, fill(4,length(ix))...) |> CuArray for ix in ixs]...,)
        @test OMEinsum.batched_contract(Val(ixs[1]), xs[1], Val(ixs[2]), xs[2], Val(iy)) |> Array ≈ loop_einsum(EinCode(ixs, iy), xs, OMEinsum.get_size_dict(ixs, xs))
        @test EinCode(ixs, iy)(xs...) |> Array ≈ loop_einsum(EinCode(ixs, iy), xs, OMEinsum.get_size_dict(ixs, xs))
    end
end


@testset "doublefloats" begin
    D = 4
    T = CuArray(rand(Double64, D, D, D, D, D, D))
    U = CuArray(rand(Double64, D, D, D))

    code = ein"abewcd,bfixgh,ajeycd,jfizgh->wxyz"
    xs = (T,T,T,T)
    M = code(xs...)
    @test M ≈ loop_einsum(code, xs, OMEinsum.get_size_dict(OMEinsum.getixs(code), xs))

    code = ein"(ubcdef,fjz),dhx,(bvghij,eiy),cgw->uvwxyz"
    _code = ein"ubcdef,fjz,dhx,bvghij,eiy,cgw->uvwxyz"
    xs = (T,U,U,T,U,U)
    M = code(xs...)
    @test M ≈ loop_einsum(_code, xs, OMEinsum.get_size_dict(OMEinsum.getixs(_code), xs))
end
