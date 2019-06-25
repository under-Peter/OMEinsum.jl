using Test
using OMEinsum
using OMEinsum: get_size_dict

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
    @test einsum(EinCode(((1,2),(2,3),(3,4)),(1,4)), (a,b,c)) ≈ a * b * c
    @test einsum(EinCode(((1,20),(20,3),(3,4)), (1,4)), (a,b,c)) ≈ a * b * c
    @test einsum(EinCode(((1,2),(2,3),(3,4)),(4,1)), (a,b,c)) ≈ permutedims(a*b*c, (2,1))
    @test einsum(EinCode(((1,2),(2,)), (1,)), (a,v)) ≈ a * v

    # contract to 0-dim array
    @test einsum(EinCode(((1,2),(1,2)), ()), (a,a))[] ≈ sum(a .* a)

    # trace
    @test einsum(EinCode(((1,1),),()), (a,))[] ≈ sum(a[i,i] for i in 1:2)
    aa = rand(2,4,4,2)
    @test einsum(EinCode(((1,2,2,1),), ()), (aa,))[] ≈ sum(aa[i,j,j,i] for i in 1:2, j in 1:4)

    # partial trace
    @test einsum(EinCode(((1,2,2,3),), (1,3)), (aa,)) ≈ sum(aa[:,i,i,:] for i in 1:4)

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

    # Hadamard product
    a = rand(2,3)
    b = rand(2,3)
    @test einsum(EinCode(((1,2),(1,2)), (1,2)), (a,b)) ≈ a .* b

    # Outer
    a = rand(2,3)
    b = rand(2,3)
    @test einsum(EinCode(((1,2),(3,4)),(1,2,3,4)),(a,b)) ≈ reshape(a,2,3,1,1) .* reshape(b,1,1,2,3)

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
    @test allocs1 < 10^5
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
    # Note: the following statement is nolonger testable, since will cause load error now!
    #@test_throws ArgumentError einsum(ein"ij,123 -> k", (a,a))
end

@testset "argument checks" begin
    @test_throws ArgumentError einsum(ein"ij,jk -> ik", (rand(2,2), rand(2,2), rand(2,2)))
    @test_throws ArgumentError einsum(ein"ij,jk,k -> ik", (rand(2,2), rand(2,2)))
    @test_throws ArgumentError einsum(ein"ij,ijk -> ik", (rand(2,2), rand(2,2)))
    @test_throws DimensionMismatch einsum(ein"ij,jk -> ik", (rand(2,3), rand(2,2)))
end
