using Test, OMEinsum, LinearAlgebra

@testset "einsum" begin
    # matrix and vector multiplication
    a,b,c = randn(2,2), rand(2,2), rand(2,2)
    v = rand(2)
    t = randn(2,2,2,2)
    @test einsum(((1,2),(2,3),(3,4)), (a,b,c)) ≈ a * b * c
    @test einsum(((1,20),(20,3),(3,4)), (a,b,c)) ≈ a * b * c
    @test einsum(((1,2),(2,3),(3,4)), (a,b,c), (4,1)) ≈ permutedims(a*b*c, (2,1))
    @test einsum(((1,2),(2,)), (a,v)) ≈ a * v

    # contract to 0-dim array
    @test einsum(((1,2),(1,2)), (a,a), ()) ≈ [sum(a .* a)]

    # trace
    @test einsum(((1,1),), (a,))[1] ≈ sum(a[i,i] for i in 1:2)
    aa = rand(2,4,4,2)
    @test einsum(((1,2,2,1),), (aa,))[1] ≈ sum(aa[i,j,j,i] for i in 1:2, j in 1:4)

    # partial trace
    @test einsum(((1,2,2,3),), (aa,)) ≈ sum(aa[:,i,i,:] for i in 1:4)

    # diag
    @test einsum(((1,2,2,3),), (aa,), (1,2,3)) ≈ permutedims(

                    reduce((x,y) -> cat(x,y, dims=3), aa[:,i,i,:] for i in 1:4),(1,3,2))

    # permutation
    @test einsum(((1,2),), (a,), (2,1)) ≈ permutedims(a,(2,1))
    @test einsum(((1,2,3,4),), (t,),(2,3,1,4)) ≈ permutedims(t,(2,3,1,4))

    # tensor contraction
    ta = zeros(size(t)[[1,2]]...)
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        ta[i,l] += t[i,j,k,l] * a[j,k]
    end
    @test einsum(((1,2,3,4), (2,3)), (t,a)) ≈  ta

    ta = zeros(size(t)[[1,2]]...)
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        ta[i,l] += t[l,k,j,i] * a[j,k]
    end
    @test einsum(((4,3,2,1), (2,3)), (t,a),(1,4)) ≈  ta

    # star-contraction
    aaa = zeros(2,2,2);
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        aaa[j,k,l] += a[i,j] * a[i,k] * a[i,l]
    end
    @test aaa ≈ einsum(((1,2),(1,3),(1,4)), (a,a,a))

    # star and contract
    aaa = zeros(2);
    for (i,j,l) in Iterators.product(1:2,1:2,1:2)
        aaa[l] += a[i,j] * a[i,j] * a[i,l]
    end
    @test einsum(((1,2),(1,2),(1,3)), (a,a,a), (3,)) ≈ aaa

    # index-sum
    a = rand(2,2,5)
    @test einsum(((1,2,3),),(a,),(1,2)) ≈ sum(a, dims=3)

    # Hadamard product
    a = rand(2,3)
    b = rand(2,3)
    @test einsum(((1,2),(1,2)), (a,b), (1,2)) ≈ a .* b

    # Outer
    a = rand(2,3)
    b = rand(2,3)
    ab = zeros(2,3,2,3)
    for (i,j,k,l) in Iterators.product(1:2,1:3,1:2,1:3)
        ab[i,j,k,l] = a[i,j] * b[k,l]
    end
    @test einsum(((1,2),(3,4)),(a,b),(1,2,3,4)) ≈ ab
end

@testset "expanding" begin
    a = randn(20,10)
    b = zeros(20,10) # i -> ii
    expandall!(b, (1,2), a, (1,2))
    @test b ≈ a

    a = randn(20,10)
    b = zeros(10,20) # i -> ii
    expandall!(b, (2,1), a, (1,2))
    @test b ≈ permutedims(a)

    a = randn(10)
    b = zeros(10,10) # i -> ii
    expandall!(b, (1,1), a, (1,))
    @test b ≈ diagm(0=>a)

    a = reshape(rand(1))
    b = zeros(10,20) # -> ij
    expandall!(b,(1,2), a, ())
    @test b ≈ fill(a[1],10,20)

    a = reshape(rand(1))
    b = zeros(20,20) # -> ii
    expandall!(b ,(2,2), a, ())
    @test b ≈ diagm(0 => fill(a[1], 20))


    a = rand(10)
    b = zeros(20,20,10) # j -> iij
    ref = zeros(20,20,10)
    for (i,j) in Iterators.product(1:20,1:10)
        ref[i,i,j] = a[j]
    end

    expandall!(b, (2,2,1), a,(1,))
    @test ref ≈ b


    a = rand(10)
    b = zeros(20,10,20) # j -> iji
    ref = zeros(20,10,20)
    for (i,j) in Iterators.product(1:20,1:10)
        ref[i,j,i] = a[j]
    end
    expandall!(b, (2,1,2), a,(1,))
    @test ref ≈ b


    a = rand(10,30)
    b = zeros(30,20,10,20) # jk -> kiji
    ref = zeros(30,20,10,20)
    for (k,i,j) in Iterators.product(1:30,1:20,1:10)
        ref[k,i,j,i] = a[j,k]
    end
    expandall!(b, (2,3,1,3), a, (1,2))
    @test ref ≈ b
end
