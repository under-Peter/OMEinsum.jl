using OMEinsum
using Test
using Einsum

@testset "OMEinsum.jl" begin
    # Write your own tests here.
    d = 5
    a = rand(fill(d,5)...)
    b = rand(fill(d,5)...)
    res = pairwise_contract((1,2,3,4,5), a, (1,6,4,5,7), b, (1,2,6,3,7))
    ref = @einsum c[i,j,l,k,m] := a[i,j,k,o,p] * b[i,l,o,p,m]
    @test ref ≈ res


    a, b, c = rand(2,3,4), rand(3,2,4), rand(2,4)
    res = einsum(((1,2,3),(2,4,3),(4,3)), (a,b,c), (1,3))
    @einsum ref[i,j] := a[i,k,j] * b[k,l,j] * c[l,j]
    @test res ≈ ref
end
