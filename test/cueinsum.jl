using Test
using OMEinsum
using CUDA, cuTENSOR
using DoubleFloats
using Zygote

CUDA.allowscalar(false)

@testset "loop einsum" begin
    a = [randn(fill(3, i)...) for i=1:4]
    ca = a .|> CuArray
    ixs = ((1,2), (2,3))
    xs = (ca[2], ca[2])
    @test loop_einsum!(ixs, (1,3), xs, zeros(3,3) |> CuArray, true, false, OMEinsum.get_size_dict(ixs, xs)) |> Array ≈ a[2]*a[2]
    @test loop_einsum!(ixs, (1,3), xs, ones(3,3) |> CuArray, 4.0, 2.0, OMEinsum.get_size_dict(ixs, xs)) |> Array ≈ 4.0 * a[2]*a[2] + 2 * ones(3, 3)
    res = 4.0 * a[2]*a[2]
    out = 2 * ones(3, 3, 3)
    for k = 1:3
        out[:,k,k] .+= res[:,k]
    end
    @test loop_einsum!(ixs, (1,3,3), xs, ones(3,3,3) |> CuArray, 4.0, 2.0, OMEinsum.get_size_dict(ixs, xs)) |> Array ≈ out
end

@testset "cuda einsum" begin
    a = [randn(fill(3, i)...) for i=1:4]
    ca = a .|> CuArray
    ixs = ((1,2), (2,3))
    xs = (ca[2], ca[2])
    @test loop_einsum!(ixs, (1,3), xs, zeros(3,3) |> CuArray, true, false, OMEinsum.get_size_dict(ixs, xs)) ≈ ca[2]*ca[2]
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
        xs = [length(ix)==0 ? CUDA.fill(1.2) : CUDA.rand(Float64, fill(D, length(ix))...) for ix in OMEinsum.getixs(code)]
        size_dict = Dict(zip(('a', 'b', 'c', 'd', 'e', 'f','i','j','k','l'), ntuple(x->D, 10)))
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
                push!(codes, OMEinsum.StaticEinCode{Char, (X1,X2),X3}())
            end
        end
    end
    for code in copy(codes)
        X1, X2 = OMEinsum.getixs(code)
        X3 = OMEinsum.getiy(code)
        push!(codes, OMEinsum.StaticEinCode{Char, ((X1...,'l'),(X2...,'l')),(X3...,'l')}())
    end

    for code in codes
        @info code
        D = 2
        xs = [length(ix)==0 ? CUDA.fill(1.2) : CUDA.rand(Float64, fill(D, length(ix))...) for ix in OMEinsum.getixs(code)]
        size_dict = Dict(zip(('a', 'b', 'c', 'd', 'e', 'f','i','j','k','l'), ntuple(x->D, 10)))
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
        xs = [length(ix)==0 ? CUDA.fill(1.2) : CUDA.rand(Float64, fill(D, length(ix))...) for ix in OMEinsum.getixs(code)]
        size_dict = Dict(zip(('a', 'b', 'c', 'd', 'e', 'f','i','j','k','l'), ntuple(x->D, 10)))
        res = einsum(code, (xs...,), size_dict)
        @test Array(res) ≈ loop_einsum(code, (Array.(xs)...,), size_dict)
        @test Array(res) ≈ Array(loop_einsum(code, (xs...,), size_dict))
    end
end

@testset "permutedims for high dimensional tensors" begin
    c = CUDA.rand(4, [rand(1:3) for _ = 2:18]...);
    @test Array(permutedims(c, 18:-1:1)) ≈ permutedims(Array(c), 18:-1:1)
end

@testset "gradient type check - CUDA" begin
    array_match(x, y) = typeof(x) == typeof(y) && size(x) == size(y)
    a = CUDA.randn(3,3)
    b = CUDA.randn(3,3)
    @test array_match(gradient(a->Array(einsum(EinCode(((1,2), (2,1)), ()), (a, b)))[] |> abs, a)[1], a)
    b = CUDA.randn(ComplexF64,3,3)
    @test array_match(gradient(a->Array(einsum(EinCode(((1,2), (2,1)), ()), (a, b)))[] |> abs, a)[1], a)
    a = CUDA.randn(ComplexF64,3,3)
    @test array_match(gradient(a->Array(einsum(EinCode(((1,2), (2,3)), ()), (a, b)))[] |> abs, a)[1], a)
    b = CUDA.randn(3,3)
    @test array_match(gradient(a->Array(einsum(EinCode(((1,2), (2,3)), ()), (a, b)))[] |> abs, a)[1], a)
end

@testset "adjoint dispatch" begin
    u = CUDA.rand(2,2); A = CUDA.rand(2,2,2);
    @test Array(ein"(ip,pql),qj -> ijl"(u', A, u)) ≈ ein"(ip,pql),qj -> ijl"(Array(CuArray(u')), Array(A), Array(u))
    @test Array(DynamicEinCode(ein"mk, ijk -> ijm")(u', A)) ≈ DynamicEinCode(ein"mk, ijk -> ijm")(Array(u'), Array(A))
    @test Array(ein"mk, ijk -> ijm"(u', A)) ≈ DynamicEinCode(ein"mk, ijk -> ijm")(Array(u'), Array(A))
end

#####################################################################
# cuTENSOR Backend Tests
#####################################################################

@testset "cuTENSOR backend - availability check" begin
    # Test backend API
    @test get_einsum_backend() isa DefaultBackend
    set_einsum_backend!(CuTensorBackend())
    @test get_einsum_backend() isa CuTensorBackend
    set_einsum_backend!(DefaultBackend())
    @test get_einsum_backend() isa DefaultBackend
end

@testset "cuTENSOR backend - binary contractions" begin
    # Skip if cuTENSOR is not available
    if cuTENSOR.has_cutensor()
        @info "Testing cuTENSOR backend (cuTENSOR available)"
        
        # Test with cuTENSOR backend
        set_einsum_backend!(CuTensorBackend())
        
        # Test different data types
        for T in [Float32, Float64, ComplexF32, ComplexF64]
            @testset "cuTENSOR $T" begin
                D = 10
                
                # Matrix multiplication: ij,jk->ik
                A = CUDA.rand(T, D, D+5)
                B = CUDA.rand(T, D+5, D+3)
                C_cutensor = ein"ij,jk->ik"(A, B)
                
                # Compare with CPU result
                A_cpu, B_cpu = Array(A), Array(B)
                C_cpu = ein"ij,jk->ik"(A_cpu, B_cpu)
                @test Array(C_cutensor) ≈ C_cpu
                
                # Batched matrix multiplication: ijl,jkl->ikl
                A = CUDA.rand(T, D, D+2, 3)
                B = CUDA.rand(T, D+2, D+1, 3)
                C_cutensor = ein"ijl,jkl->ikl"(A, B)
                C_cpu = ein"ijl,jkl->ikl"(Array(A), Array(B))
                @test Array(C_cutensor) ≈ C_cpu
                
                # Trace contraction: ij,ji->
                A = CUDA.rand(T, D, D)
                B = CUDA.rand(T, D, D)
                C_cutensor = ein"ij,ji->"(A, B)
                C_cpu = ein"ij,ji->"(Array(A), Array(B))
                @test Array(C_cutensor)[] ≈ C_cpu[]
                
                # Outer product: i,j->ij
                A = CUDA.rand(T, D)
                B = CUDA.rand(T, D+2)
                C_cutensor = ein"i,j->ij"(A, B)
                C_cpu = ein"i,j->ij"(Array(A), Array(B))
                @test Array(C_cutensor) ≈ C_cpu
                
                # General contraction: ijk,jkl->il
                A = CUDA.rand(T, 5, 6, 7)
                B = CUDA.rand(T, 6, 7, 8)
                C_cutensor = ein"ijk,jkl->il"(A, B)
                C_cpu = ein"ijk,jkl->il"(Array(A), Array(B))
                @test Array(C_cutensor) ≈ C_cpu
                
                # Transposed output: ij,jk->ki
                A = CUDA.rand(T, D, D+2)
                B = CUDA.rand(T, D+2, D+3)
                C_cutensor = ein"ij,jk->ki"(A, B)
                C_cpu = ein"ij,jk->ki"(Array(A), Array(B))
                @test Array(C_cutensor) ≈ C_cpu
            end
        end
        
        # Reset backend
        set_einsum_backend!(DefaultBackend())
    else
        @info "Skipping cuTENSOR tests (cuTENSOR not available)"
        @test_skip "cuTENSOR not available"
    end
end

@testset "cuTENSOR backend - comparison with DefaultBackend" begin
    if cuTENSOR.has_cutensor()
        @info "Comparing cuTENSOR vs DefaultBackend results"
        
        for T in [Float32, Float64]
            @testset "Comparison $T" begin
                D = 32
                
                # Test various contraction patterns
                test_cases = [
                    (ein"ij,jk->ik", (D, D+5), (D+5, D+3)),           # matmul
                    (ein"ijk,jkl->il", (8, 10, 12), (10, 12, 14)),    # tensor contraction
                    (ein"ijl,jkl->ikl", (8, 10, 5), (10, 12, 5)),     # batched
                    (ein"ij,kj->ik", (D, D+2), (D+3, D+2)),           # transposed B
                    (ein"ji,jk->ik", (D+2, D), (D+2, D+3)),           # transposed A
                ]
                
                for (code, size1, size2) in test_cases
                    A = CUDA.rand(T, size1...)
                    B = CUDA.rand(T, size2...)
                    
                    # DefaultBackend result
                    set_einsum_backend!(DefaultBackend())
                    C_default = code(A, B)
                    
                    # CuTensorBackend result
                    set_einsum_backend!(CuTensorBackend())
                    C_cutensor = code(A, B)
                    
                    @test Array(C_cutensor) ≈ Array(C_default) rtol=1e-5
                end
            end
        end
        
        set_einsum_backend!(DefaultBackend())
    else
        @test_skip "cuTENSOR not available"
    end
end

@testset "cuTENSOR backend - scaling factors" begin
    if cuTENSOR.has_cutensor()
        set_einsum_backend!(CuTensorBackend())
        
        D = 16
        A = CUDA.rand(Float64, D, D)
        B = CUDA.rand(Float64, D, D)
        C = CUDA.rand(Float64, D, D)
        
        code = EinCode((('i','j'),('j','k')),('i','k'))
        
        # Test with sx=2.0, sy=0.5
        C_result = copy(C)
        einsum!(code, (A, B), C_result, 2.0, 0.5)
        
        A_cpu, B_cpu, C_cpu = Array(A), Array(B), Array(C)
        C_expected = 2.0 * (A_cpu * B_cpu) + 0.5 * C_cpu
        
        @test Array(C_result) ≈ C_expected rtol=1e-10
        
        set_einsum_backend!(DefaultBackend())
    else
        @test_skip "cuTENSOR not available"
    end
end

@testset "cuTENSOR backend - nested einsum" begin
    if cuTENSOR.has_cutensor()
        set_einsum_backend!(CuTensorBackend())
        
        # Test nested contraction
        D = 8
        A = CUDA.rand(Float32, D, D)
        B = CUDA.rand(Float32, D, D)
        C = CUDA.rand(Float32, D, D)
        
        code = ein"(ij,jk),kl->il"
        result_cutensor = code(A, B, C)
        
        # Compare with CPU
        result_cpu = code(Array(A), Array(B), Array(C))
        @test Array(result_cutensor) ≈ result_cpu rtol=1e-4
        
        set_einsum_backend!(DefaultBackend())
    else
        @test_skip "cuTENSOR not available"
    end
end

@testset "cuTENSOR backend - Float16" begin
    if cuTENSOR.has_cutensor()
        set_einsum_backend!(CuTensorBackend())
        
        D = 32
        A = CUDA.rand(Float16, D, D)
        B = CUDA.rand(Float16, D, D)
        
        C_cutensor = ein"ij,jk->ik"(A, B)
        
        # Compare with Float32 computation for accuracy reference
        A32 = Float32.(Array(A))
        B32 = Float32.(Array(B))
        C_ref = ein"ij,jk->ik"(A32, B32)
        
        @test Float32.(Array(C_cutensor)) ≈ C_ref rtol=1e-2
        
        set_einsum_backend!(DefaultBackend())
    else
        @test_skip "cuTENSOR not available"
    end
end