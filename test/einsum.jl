using Test, OMEinsum

@testset "einsum" begin
    # Write your own tests here.
    a,b,c = randn(2,2), rand(2,2), rand(2,2)
    v = rand(2)
    t = randn(2,2,2,2)
    @test einsum(((1,2),(2,3),(3,4)), (a,b,c)) ≈ a * b * c
    @test einsum(((1,20),(20,3),(3,4)), (a,b,c)) ≈ a * b * c
    @test einsum(((1,2),(2,3),(3,4)), (a,b,c), (4,1)) ≈ permutedims(a*b*c, (2,1))
    @test einsum(((1,2),(2,)), (a,v)) ≈ a * v

    @test_broken einsum(((1,1),), (a,))
    @test einsum(((1,2),), (a,), (2,1)) ≈ permutedims(a,(2,1))
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

    aaa = zeros(2,2,2);
    for (i,j,k,l) in Iterators.product(1:2,1:2,1:2,1:2)
        aaa[j,k,l] += a[i,j] * a[i,k] * a[i,l]
    end
    @test aaa ≈ einsum(((1,2),(1,3),(1,4)), (a,a,a))

    let p = (2,3,1,4)
        @test einsum(((1,2,3,4),), (t,),p) ≈ permutedims(t,p)
    end
end
