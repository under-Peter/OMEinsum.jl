using Test
using OMEinsum
using AMDGPU
using DoubleFloats
using Zygote

AMDGPU.allowscalar(false)

@testset "loop einsum" begin
    a = [randn(fill(3, i)...) for i = 1:4]
    ca = a .|> ROCArray
    ixs = ((1, 2), (2, 3))
    xs = (ca[2], ca[2])
    @test loop_einsum!(ixs, (1, 3), xs, zeros(3, 3) |> ROCArray, true, false, OMEinsum.get_size_dict(ixs, xs)) |> Array ≈ a[2] * a[2]
    @test loop_einsum!(ixs, (1, 3), xs, ones(3, 3) |> ROCArray, 4.0, 2.0, OMEinsum.get_size_dict(ixs, xs)) |> Array ≈ 4.0 * a[2] * a[2] + 2 * ones(3, 3)
    res = 4.0 * a[2] * a[2]
    out = 2 * ones(3, 3, 3)
    for k = 1:3
        out[:, k, k] .+= res[:, k]
    end
    @test loop_einsum!(ixs, (1, 3, 3), xs, ones(3, 3, 3) |> ROCArray, 4.0, 2.0, OMEinsum.get_size_dict(ixs, xs)) |> Array ≈ out
end

@testset "roc einsum" begin
    a = [randn(fill(3, i)...) for i = 1:4]
    ca = a .|> ROCArray
    ixs = ((1, 2), (2, 3))
    xs = (ca[2], ca[2])
    @test loop_einsum!(ixs, (1, 3), xs, zeros(3, 3) |> ROCArray, true, false, OMEinsum.get_size_dict(ixs, xs)) ≈ ca[2] * ca[2]
    for f in [ein"ij,jk->ik", ein"ii->", ein"ijj ->i", ein"ij,ik,il->jkl", ein"ii->i", ein"ijl->i", ein"i->ii", ein"ij,jk,kl->il", ein"ij,ij,ij -> ij"]
        cins = map(ix -> ca[length(ix)], OMEinsum.getixs(f))
        ins = map(ix -> a[length(ix)], OMEinsum.getixs(f))
        @test f(cins...) isa DenseROCArray
        @test loop_einsum(f, cins, OMEinsum.get_size_dict(OMEinsum.getixs(f), cins)) |> Array ≈ f(ins...)
        @test f(ins...) ≈ Array(f(cins...))
    end
end

@testset "fallback - getindex IR error" begin
    a = rand(ComplexF64, 2, 2, 2)
    ca = ROCArray(a)
    @test Array(ein"npu,por,dom,lmn -> urdl"(ca, ca, ca, ca)) ≈ ein"npu,por,dom,lmn -> urdl"(a, a, a, a)
end

@testset "isbatchmul" begin
    for (ixs, iy) in [(((1, 2), (2, 3)), (1, 3)), (((1, 2, 3), (2, 3)), (1, 3)),
        (((7, 1, 2, 3), (2, 4, 3, 7)), (1, 4, 3)),
        (((3,), (3,)), (3,)), (((3, 1), (3,)), (3, 1))
    ]
        xs = ([randn(ComplexF64, fill(4, length(ix))...) |> ROCArray for ix in ixs]...,)
        @test EinCode(ixs, iy)(xs...) |> Array ≈ loop_einsum(EinCode(ixs, iy), xs, OMEinsum.get_size_dict(ixs, xs)) |> Array
    end
end


@testset "doublefloats" begin
    D = 2
    T = ROCArray(rand(Double64, D, D, D, D, D, D))
    U = ROCArray(rand(Double64, D, D, D))

    code = ein"abewcd,bfixgh,ajeycd,jfizgh->wxyz"
    xs = (T, T, T, T)
    M = code(xs...)
    @test M |> Array ≈ loop_einsum(code, xs, OMEinsum.get_size_dict(OMEinsum.getixs(code), xs)) |> Array

    code = ein"(ubcdef,fjz),dhx,(bvghij,eiy),cgw->uvwxyz"
    _code = ein"ubcdef,fjz,dhx,bvghij,eiy,cgw->uvwxyz"
    xs = (T, U, U, T, U, U)
    M = code(xs...)
    # mapreducedim! calls to dynamic tuple splatting.
    @test M |> Array ≈ loop_einsum(_code, xs, OMEinsum.get_size_dict(OMEinsum.getixs(_code), xs)) |> Array
end

@testset "unary einsum rules" begin
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
    ]
        @info code
        D = 2
        xs = [length(ix) == 0 ? AMDGPU.fill(1.2) : AMDGPU.rand(Float64, fill(D, length(ix))...) for ix in OMEinsum.getixs(code)]
        size_dict = Dict(zip(('a', 'b', 'c', 'd', 'e', 'f', 'i', 'j', 'k', 'l'), ntuple(x -> D, 10)))
        res = einsum(code, (xs...,), size_dict)
        @test Array(res) ≈ loop_einsum(code, (Array.(xs)...,), size_dict)
        @test Array(res) ≈ Array(loop_einsum(code, (xs...,), size_dict))
    end
end

@testset "binary einsum rules" begin
    codes = [
        # binary
        ein",->",
        ein"i,->i",
        ein"j,j->",
        ein",k->k",
        ein"j,jk->k",
        ein"j,kj->k",
        ein"ij,j->i",
        ein"ji,j->i",
        ein"i,k->ik",
        ein"i,k->ki",
    ]

    for (i1, X1) in enumerate([('i', 'j'), ('j', 'i')])
        for (i2, X2) in enumerate([('j', 'k'), ('k', 'j')])
            for (i3, X3) in enumerate([('i', 'k'), ('k', 'i')])
                push!(codes, OMEinsum.StaticEinCode{Char,(X1, X2),X3}())
            end
        end
    end
    for code in copy(codes)
        X1, X2 = OMEinsum.getixs(code)
        X3 = OMEinsum.getiy(code)
        push!(codes, OMEinsum.StaticEinCode{Char,((X1..., 'l'), (X2..., 'l')),(X3..., 'l')}())
    end

    for code in codes
        @info code
        D = 2
        xs = [length(ix) == 0 ? AMDGPU.fill(1.2) : AMDGPU.rand(Float64, fill(D, length(ix))...) for ix in OMEinsum.getixs(code)]
        size_dict = Dict(zip(('a', 'b', 'c', 'd', 'e', 'f', 'i', 'j', 'k', 'l'), ntuple(x -> D, 10)))
        res = einsum(code, (xs...,), size_dict)
        @test Array(res) ≈ loop_einsum(code, (Array.(xs)...,), size_dict)
        @test Array(res) ≈ Array(loop_einsum(code, (xs...,), size_dict))
    end
end

@testset "composite einsum" begin
    for code in [
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
        xs = [length(ix) == 0 ? AMDGPU.fill(1.2) : AMDGPU.rand(Float64, fill(D, length(ix))...) for ix in OMEinsum.getixs(code)]
        size_dict = Dict(zip(('a', 'b', 'c', 'd', 'e', 'f', 'i', 'j', 'k', 'l'), ntuple(x -> D, 10)))
        res = einsum(code, (xs...,), size_dict)
        @test Array(res) ≈ loop_einsum(code, (Array.(xs)...,), size_dict)
        @test Array(res) ≈ Array(loop_einsum(code, (xs...,), size_dict))
    end
end

@testset "permutedims for high dimensional tensors" begin
    c = AMDGPU.rand(4, [rand(1:3) for _ = 2:18]...)
    @test Array(permutedims(c, 18:-1:1)) ≈ permutedims(Array(c), 18:-1:1)
end

@testset "gradient type check - AMDGPU" begin
    array_match(x, y) = typeof(x) == typeof(y) && size(x) == size(y)
    a = AMDGPU.randn(3, 3)
    b = AMDGPU.randn(3, 3)
    @test array_match(gradient(a -> Array(einsum(EinCode(((1, 2), (2, 1)), ()), (a, b)))[] |> abs, a)[1], a)
    b = randn(ComplexF64, 3, 3) |> ROCArray
    @test array_match(gradient(a -> Array(einsum(EinCode(((1, 2), (2, 1)), ()), (a, b)))[] |> abs, a)[1], a)
    a = randn(ComplexF64, 3, 3) |> ROCArray
    @test array_match(gradient(a -> Array(einsum(EinCode(((1, 2), (2, 3)), ()), (a, b)))[] |> abs, a)[1], a)
    b = AMDGPU.randn(3, 3)
    @test array_match(gradient(a -> Array(einsum(EinCode(((1, 2), (2, 3)), ()), (a, b)))[] |> abs, a)[1], a)
end

@testset "adjoint dispatch" begin
    u = AMDGPU.rand(2, 2)
    A = AMDGPU.rand(2, 2, 2)
    @test Array(ein"(ip,pql),qj -> ijl"(u', A, u)) ≈ ein"(ip,pql),qj -> ijl"(Array(ROCArray(u')), Array(A), Array(u))
    @test Array(DynamicEinCode(ein"mk, ijk -> ijm")(u', A)) ≈ DynamicEinCode(ein"mk, ijk -> ijm")(Array(u'), Array(A))
    @test Array(ein"mk, ijk -> ijm"(u', A)) ≈ DynamicEinCode(ein"mk, ijk -> ijm")(Array(u'), Array(A))
end