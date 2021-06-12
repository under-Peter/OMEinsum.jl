using Test
using OMEinsum
using CUDA
using DoubleFloats

CUDA.allowscalar(false)

@testset "cuda einsum" begin
    a = [randn(fill(3, i)...) for i=1:4]
    ca = a .|> CuArray
    ixs = ((1,2), (2,3))
    xs = (ca[2], ca[2])
    @test loop_einsum!(EinCode(ixs, (1,3)), xs, zeros(3,3) |> CuArray, OMEinsum.get_size_dict(ixs, xs)) ≈ ca[2]*ca[2]
    for f in [ein"ij,jk->ik", ein"ii->", ein"ijj ->i", ein"ij,ik,il->jkl", ein"ii->i", ein"ijl->i", ein"i->ii", ein"ij,jk,kl->il", ein"ij,ij,ij -> ij"]
        cins = map(ix->ca[length(ix)], OMEinsum.getixs(f))
        ins = map(ix->a[length(ix)], OMEinsum.getixs(f))
        @test f(cins...) isa DenseCuArray
        @test loop_einsum(f, cins, OMEinsum.get_size_dict(OMEinsum.getixs(f), cins)) |> Array ≈ f(ins...)
        @test f(ins...) ≈ Array(f(cins...))
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
        @test EinCode(ixs, iy)(xs...) |> Array ≈ loop_einsum(EinCode(ixs, iy), xs, OMEinsum.get_size_dict(ixs, xs)) |> Array
    end
end


@testset "doublefloats" begin
    D = 2
    T = CuArray(rand(Double64, D, D, D, D, D, D))
    U = CuArray(rand(Double64, D, D, D))

    code = ein"abewcd,bfixgh,ajeycd,jfizgh->wxyz"
    xs = (T,T,T,T)
    M = code(xs...)
    @test M |> Array ≈ loop_einsum(code, xs, OMEinsum.get_size_dict(OMEinsum.getixs(code), xs)) |> Array

    code = ein"(ubcdef,fjz),dhx,(bvghij,eiy),cgw->uvwxyz"
    _code = ein"ubcdef,fjz,dhx,bvghij,eiy,cgw->uvwxyz"
    xs = (T,U,U,T,U,U)
    M = code(xs...)
    # mapreducedim! calls to dynamic tuple splatting.
    @test_broken M |> Array ≈ loop_einsum(_code, xs, OMEinsum.get_size_dict(OMEinsum.getixs(_code), xs)) |> Array
end

@testset "binary einsum" begin
    for code in [
            ein"ij->",  # sum
            ein"ij->j", # sum
            ein"ii->",  # tr
            ein"ii->i",  # diag
            ein"ijk->kij",   # permutedims
            ein"a->aaaa",   # ~diag
            ein"ac->acc",   # ~diag
            ein"->ii",   # ~tr
            ein"i->ik",   # ~sum
            ein"->ik",   # ~sum
            ein"illljkk->kijjcc",   # general
            # binary
            ein",->",
            ein"ijl,jl->il",
            ein"ab,bc->ac",
            ein"abb,bc->ac",  # with diag in
            ein"ab,bc->acc",  # with diag out
            ein"ab,bce->ac",  # with sum in
            ein"ab,bc->ace",  # with sum out
            ein"bal,bcl->lcae",  # with perm in
            ein"bal,bcl->ca",  # with multi-edge in
            ein"bal,bc->lca",  # with multi-edge out
            ein"ddebal,bcf->lcac",  # with all
        ]
        @info code
        D = 2
        xs = [length(ix)==0 ? CUDA.fill(1.2) : CUDA.rand(Float64, fill(D, length(ix))...) for ix in OMEinsum.getixs(code)]
        size_dict = Dict(zip(('a', 'b', 'c', 'd', 'e', 'f','i','j','k','l'), ntuple(x->D, 10)))
        res = einsum(code, (xs...,), size_dict)
        @test Array(res) ≈ loop_einsum(code, (Array.(xs)...,), size_dict)
        @test Array(res) ≈ Array(loop_einsum(code, (xs...,), size_dict))
    end
end

@testset "binary rules" begin
    for (code, a, b) in [
        (ein"j,j->", randn(10), randn(10)),
        (ein"i,->i", randn(10), fill(2.0, ())),
        (ein",->", fill(2.0,()), fill(2.0, ())),
        (ein"il,kl->ikl", randn(10, 10), randn(10, 10)),
        ]
        res0 = code(a, b)
        res1 = code(CuArray(a), CuArray(b))
        @test res1 isa CuArray
        @test res0 ≈ Array(res1)
    end
end