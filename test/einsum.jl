using Test
using OMEinsum
using OMEinsum: get_size_dict
using SymEngine
using LinearAlgebra: I, tr

SymEngine.free_symbols(syms::Union{Real, Complex}) = Basic[]
SymEngine.free_symbols(syms::AbstractArray{T}) where {T<:Union{Real, Complex}} = Basic[]
function rand_assign(syms...)
    fs = union(free_symbols.(syms)...)
    Dict(zip(fs, randn(length(fs))))
end

function _basic_approx(x, y; atol=1e-8)
    diff = x-y
    assign = rand_assign(x, y)
    length(assign) > 0 && (diff = subs.(diff, Ref.((assign...,))...))
    nres = ComplexF64.(diff)
    all(isapprox.(nres, 0; atol=atol))
end

Base.:≈(x::AbstractArray{<:Basic}, y::AbstractArray; atol=1e-8) = _basic_approx(x, y, atol=atol)
Base.:≈(x::AbstractArray, y::AbstractArray{<:Basic}; atol=1e-8) = _basic_approx(x, y, atol=atol)
Base.:≈(x::AbstractArray{<:Basic}, y::AbstractArray{<:Basic}; atol=1e-8) = _basic_approx(x, y, atol=atol)
Base.Complex{T}(a::Basic) where T = T(real(a)) + im*T(imag(a))

@testset "unary einsum" begin
    size_dict = Dict(1=>3,2=>3,3=>3,4=>4,5=>5)
    ix = (1,2,3,3,4)
    x = randn(3,3,3,3,4)
    iy = (3,5,1,1,2,5)
    y = randn(3,5,3,3,3,5)
    # Diag, Sum, Repeat, Duplicate
    @test einsum!((ix,), iy, (x,), y, true, false, size_dict) ≈ loop_einsum(EinCode((ix,), iy), (x,), size_dict)
    ix = (1,2,3,4)
    x = randn(3,3,3,4)
    iy = (4,3,1,2)
    y = randn(4,3,3,3)
    # Permutedims
    @test einsum!((ix,), iy, (x,), y, true, false, size_dict) ≈ loop_einsum(EinCode((ix,), iy), (x,), size_dict)
    # None
    ix = (1,2,3,4)
    x = randn(3,3,3,4)
    iy = (1,2,3,4)
    y = randn(3,3,3,4)
    @test einsum!((ix,), iy, (x,), y, true, false, size_dict) ≈ loop_einsum(EinCode((ix,), iy), (x,), size_dict)
    # tr
    ix = (1,1)
    x = randn(3,3)
    iy = ()
    y = fill(1.0)
    @test einsum!((ix,), iy, (x,), y, true, false, size_dict)[] ≈ tr(x)
end

@testset "binary einsum" begin
    size_dict = Dict(1=>3,2=>3,3=>3,4=>4,5=>5)
    ix = (1,2,3,3,4)
    x = randn(3,3,3,3,4)
    iy = (3,5,1,1,2,5)
    y = randn(3,5,3,3,3,5)
    iz = (1,2,3,4,5,5)
    z = randn(3,3,3,4,5,5)
    @test einsum!((ix, iy), iz, (x, y), z, true, false, size_dict) ≈ loop_einsum(EinCode((ix, iy), iz), (x, y), size_dict)
    @test einsum!((ix, iy), iz, (x, y), copy(z), 5.0, 3.0, size_dict) ≈ loop_einsum!((ix, iy), iz, (x, y), copy(z), 5.0, 3.0, size_dict)
    @test einsum!((ix, iy), iz, (x, y), copy(z), 5.0, 1.0, size_dict) ≈ loop_einsum!((ix, iy), iz, (x, y), copy(z), 5.0, 1.0, size_dict)
end

@testset "nary, einsum" begin
     size_dict = Dict(1=>3,2=>3,3=>3,4=>4,5=>5)
    ix = (1,2,3,3,4)
    x = randn(3,3,3,3,4)
    iy = (3,5,1)
    y = randn(3,5,3)
    iz = (1,2,3,4,5,5)
    z = randn(3,3,3,4,5,5)
    @test einsum!((ix, iy, iz), (), (x, y, z), fill(1.0), true, false, size_dict) ≈ loop_einsum(EinCode((ix, iy, iz), ()), (x, y, z), size_dict)
end

@testset "get output array" begin
    xs = (randn(4,4), randn(3))
    @test OMEinsum.get_output_array(xs, (5, 5), false) isa Array{Float64}
    xs = (randn(4,4), randn(ComplexF64, 3))
    @test OMEinsum.get_output_array(xs, (5, 5), false) isa Array{ComplexF64}
end

@testset "tensor order check" begin
    ixs = ((1,2), (2,3))
    a = randn(3,3)
    @test_throws ArgumentError get_size_dict(ixs, (a,a,a))
    a = randn(3,3,3)
    @test_throws ArgumentError get_size_dict(ixs, (a,a))
end

@testset "get size dict" begin
    a = randn(3,2)
    sizedict = get_size_dict(((1,2), (2,3)), (a, a'))
    @test (sizedict[1], sizedict[2], sizedict[3]) == (3,2,3)
    sizedict = get_size_dict((('i','j'), ('k','j')), (a, a))
    @test (sizedict['i'], sizedict['j'], sizedict['k']) == (3,2,3)
    @test_throws DimensionMismatch get_size_dict((('i','j'), ('j','k')), (a, a))
end

@testset "einsum" begin
    # matrix and vector multiplication
    a,b,c = randn(2,2), rand(2,2), rand(2,2)
    v = rand(2)
    t = randn(2,2,2,2)
    @test einsum(ein"ijkl -> ijkl", (t,)) ≈ t
    @test einsum(ein"αβγδ -> αβγδ", (t,)) ≈ t
    @test einsum(EinCode(((1,2),(2,3),(3,4)),(1,4)), (a,b,c)) ≈ a * b * c
    @test einsum(EinCode(((1,20),(20,3),(3,4)), (1,4)), (a,b,c)) ≈ a * b * c
    @test einsum(EinCode(((1,2),(2,3),(3,4)),(4,1)), (a,b,c)) ≈ permutedims(a*b*c, (2,1))
    @test einsum(EinCode(((1,2),(2,)), (1,)), (a,v)) ≈ a * v

    # more matmul
    @test ein"ij,jk -> ik"(a,a) ≈ a * a
    @test ein"ij,jk -> ki"(a,a) ≈ transpose(a * a)
    @test ein"ij,kj -> ik"(a,a) ≈ a * transpose(a)
    @test ein"ij,kj -> ki"(a,a) ≈ transpose(a * transpose(a))
    @test ein"ji,jk -> ik"(a,a) ≈ transpose(a) * a
    @test ein"ji,jk -> ki"(a,a) ≈ transpose(transpose(a) * a)
    @test ein"ji,kj -> ik"(a,a) ≈ transpose(a) * transpose(a)
    @test ein"ji,kj -> ki"(a,a) ≈ transpose(transpose(a) * transpose(a))

    # contract to 0-dim array
    @test einsum(EinCode(((1,2),(1,2)), ()), (a,a))[] ≈ sum(a .* a)

    # trace
    @test einsum(EinCode(((1,1),),()), (a,))[] ≈ sum(a[i,i] for i in 1:2)
    @test einsum(EinCode(((1,1),),()), (a,)) isa Array
    aa = rand(2,4,4,2)
    @test einsum(EinCode(((1,2,2,1),), ()), (aa,))[] ≈ sum(aa[i,j,j,i] for i in 1:2, j in 1:4)

    # partial trace
    @test einsum(EinCode(((1,2,2,3),), (1,3)), (aa,)) ≈ sum(aa[:,i,i,:] for i in 1:4)
    # with permutation
    @test einsum(EinCode(((1,2,2,3),), (3,1)), (aa,)) ≈ transpose(sum(aa[:,i,i,:] for i in 1:4))

    # diag
    @test einsum(EinCode(((1,2,2,3),), (1,2,3)), (aa,)) ≈ aa[:,[CartesianIndex(i,i) for i in 1:4],:]

    # permutation
    @test einsum(EinCode(((1,2),), (2,1)), (a,)) ≈ permutedims(a,(2,1))
    @test einsum(EinCode(((1,2,3,4),),(2,3,1,4)), (t,)) ≈ permutedims(t,(2,3,1,4))

    # tensor contraction
    ta = zeros(size(t)[[1,2]]...)
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        ta[i,l] += t[i,j,k,l] * a[j,k]
    end
    @test einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a)) ≈  ta

    ta = zeros(size(t)[[1,2]]...)
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        ta[i,l] += t[l,k,j,i] * a[j,k]
    end
    @test einsum(EinCode(((4,3,2,1), (2,3)),(1,4)), (t,a)) ≈  ta

    # star-contraction
    aaa = zeros(2,2,2);
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        aaa[j,k,l] += a[i,j] * a[i,k] * a[i,l]
    end
    @test aaa ≈ einsum(EinCode(((1,2),(1,3),(1,4)),(2,3,4)), (a,a,a))

    # star and contract
    aaa = zeros(2);
    for (i,j,l) in Iterators.product(1:2,1:2,1:2)
        aaa[l] += a[i,j] * a[i,j] * a[i,l]
    end
    @test einsum(EinCode(((1,2),(1,2),(1,3)), (3,)), (a,a,a)) ≈ aaa

    # index-sum
    a = rand(2,2,5)
    @test einsum(EinCode(((1,2,3),),(1,2)),(a,)) ≈ sum(a, dims=3)
    # with permutation
    @test ein"ijk -> ki"(a) ≈ transpose(dropdims(sum(a,dims=2),dims=2))
    @test ein"ijk -> "(a)[] ≈ sum(a)

    # Hadamard product
    a = rand(2,3)
    b = rand(2,3)
    @test einsum(EinCode(((1,2),(1,2)), (1,2)), (a,b)) ≈ a .* b

    # Outer
    a = rand(2,3)
    b = rand(2,3)
    @test einsum(EinCode(((1,2),(3,4)),(1,2,3,4)),(a,b)) ≈ reshape(a,2,3,1,1) .* reshape(b,1,1,2,3)

    # Broadcasting
    @test ein"->ii"(OMEinsum.asarray(1); size_info=Dict('i'=>5)) == Matrix(I, 5, 5)

    # trivil
    @test ein"->ii"(asarray(1); size_info=Dict('i'=>5)) == Matrix(I, 5, 5)
    res = ein"->"(asarray(3))
    @test res isa Array
    @test res[] === 3
    res = ein",->"(asarray(3), asarray(4))
    @test res isa Array
    @test res[] === 12

    # Projecting to diag
    a = rand(2,2)
    a2 = [a[1] 0; 0 a[4]]
    @test einsum(EinCode(((1,1),), (1,1)), (a,)) ≈ a2

    ## operations that can be combined
    a = rand(2,2,2,2)
    @test einsum(EinCode(((1,1,2,2),), ()), (a,))[] ≈ sum(a[[CartesianIndex(i,i) for i in 1:2], [CartesianIndex(i,i) for i in 1:2]])

    @test einsum(EinCode(((1,2,3,4), (3,4,5,6)), (1,2,5,6)), (a,a)) ≈ reshape(reshape(a,4,4) * reshape(a,4,4),2,2,2,2)
end

@testset "fallback" begin
    # while we expect some scaling in the allocations for multiple inputs, it
    # shouldn't increase too much
    a = rand(100,100)
    b = rand(100,100)
    einsum(EinCode(((1,2),(2,3)),(1,3)), (a,b))
    allocs1 = @allocated einsum(EinCode(((1,2),(2,3)),(1,3)), (a,b))
    @test allocs1 < 1.2e5
    einsum(EinCode(((1,2),(2,3),(3,4)),(1,4)), (a,b,b))
    allocs2 = @allocated einsum(EinCode(((1,2),(2,3),(3,4)),(1,4)), (a,b,b))
    # doing twice the work (two multiplications instead of one) shouldn't
    # incure much more than twice the allocations.
    @test allocs2 < 2.1 * allocs1

    @test_throws MethodError einsum(((1,2),(2,3)), (a,a))
end

@testset "error handling" begin
    a = randn(3,3)
    b = randn(4,4)
    @test_throws DimensionMismatch einsum(EinCode(((1,2), (2,3)), (1,3)), (a, b))
end

@testset "string input" begin
    a = randn(3,3)
    @test einsum(ein"ij,jk -> ik", (a,a)) ≈ einsum(EinCode(((1,2),(2,3)), (1,3)), (a,a))
    @test ein"ij,jk -> ik"(a,a) ≈ einsum(EinCode(((1,2),(2,3)), (1,3)), (a,a))
    @test ein"αβ,βγ -> αγ"(a,a) ≈ a * a
    # Note: the following statement is nolonger testable, since will cause load error now!
    #@test_throws ArgumentError einsum(ein"ij,123 -> k", (a,a))
end

@testset "non inplace macro input" begin
    a = randn(2,2)
    @test a * a ≈ @ein [i,k] := a[i,j] * a[j,k]
    @test sum(a[i,i] for i in 1:2) ≈ (@ein [] := a[i,i])[]
    @test [a[1,1] 0; 0 a[2,2]] ≈ @ein [i,i] := a[i,i]
    @test permutedims(a) ≈ @ein [α,a] := a[a,α]
    @test permutedims(a) ≈ @ein [α,1] := a[1,α]
end

@testset "inplace macro input" begin
    a = randn(2,2)
    b = randn(2,2)
    c = randn(2,2)
    t = randn(2,2)
    cc = copy(c)
    @ein! t[i,k] := a[i,j] * b[j,k]
    @ein! c[i,k] += a[i,j] * b[j,k]
    @test a * b ≈ t
    @test cc + a * b ≈ c
end

@testset "argument checks" begin
    @test_throws ArgumentError einsum(ein"ij,jk -> ik", (rand(2,2), rand(2,2), rand(2,2)))
    @test_throws ArgumentError einsum(ein"ij,jk,k -> ik", (rand(2,2), rand(2,2)))
    @test_throws ArgumentError einsum(ein"ij,ijk -> ik", (rand(2,2), rand(2,2)))
    @test_throws DimensionMismatch einsum(ein"ij,jk -> ik", (rand(2,3), rand(2,2)))
end

@testset "isbatchmul" begin
    for (ixs, iy) in [(((1,2), (2,3)), (1,3)), (((1,2,3), (2,3)), (1,3)),
                        (((7,1,2,3), (2,4,3,7)), (1,4,3)),
                        (((3,), (3,)), (3,)), (((3,1), (3,)), (3,1))
                        ]
        xs = ([randn(ComplexF64, fill(4,length(ix))...) for ix in ixs]...,)
        @test EinCode(ixs, iy)(xs...) ≈ loop_einsum(EinCode(ixs, iy), xs, OMEinsum.get_size_dict(ixs, xs))
    end
end

@testset "issue 136" begin
    @test EinCode(((1,2,3),(2,)),(1,3))(ones(2,2,1), ones(2)) == reshape([2,2.0], 2, 1)
    @test EinCode(((1,2,3),(2,)),(1,3))(ones(2,2,0), ones(2)) == reshape(zeros(0), 2, 0)
end

@testset "fix rule cc,cb->bc" begin
    size_dict = Dict('a'=>2,'b'=>2,'c'=>2)
    for code in [ein"c,c->cc", ein"c,cc->c", ein"cc,c->cc", ein"cc,cc->cc", ein"cc,cb->bc", ein"cb,bc->cc", ein"ac,cc->ac"]
        @info code
        a = randn(fill(2, length(getixsv(code)[1]))...)
        b = randn(fill(2, length(getixsv(code)[2]))...)
        @test code(a, b) ≈ OMEinsum.loop_einsum(code, (a,b), size_dict)
    end
end

# patch for SymEngine
Base.promote_rule(::Type{Bool}, ::Type{Basic}) = Basic
@testset "allow loops" begin
    t = rand(5,5,5,5)
    a = rand(5,5)
    size_dict = Dict(zip((1,2,3,4,2,3), ((size(t)..., size(a)...))))

    OMEinsum.allow_loops(false)
    @test_throws ErrorException loop_einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict)
    OMEinsum.allow_loops(true)

    ta = loop_einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict)
    @test einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict) ≈  ta

    # index-sum
    t = Basic.(rand(5,5,5,5))
    a = Basic.(rand(5,5))
    size_dict = Dict(zip((1,2,3,4,2,3), ((size(t)..., size(a)...))))
    ta = loop_einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict)
    @test einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a), size_dict) ≈  ta
end

@testset "symbolic" begin
    @test ein"ij->ijij"([1 1; 1 exp(im*Basic(π)/2)]) == [1 0; 0 0;;; 0 0; 1 0;;;; 0 1; 0 0;;; 0 0; 0 exp(im*Basic(π)/2)]
end
