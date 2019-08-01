include("reduce_einsum.jl")

using Test
m = randn(100,50)
cm = m |> CuArray
using CUDAdrv
CUDAdrv.@profile (CuArrays.@sync ein"ij,ik,il->jkl"(cm,cm,cm))

ein"ij,ik,il->jkl"(m,m,m)
ein"ij,ik,il->jkl"(m,m,m) ≈ Array(ein"ij,ik,il->jkl"(cm,cm,cm))

# task: accelerate the following code by a factor of 50 (close to pytorch performance then).
#@benchmark (CuArrays.@sync ein"ij,ik,il->jkl"($cm, $cm, $cm)) seconds=1

@testset "cuda array" begin
    a = randn(Float32, 100, 50)
    ca = CuArray(a)
    @test maximum(abs.(Array(ein"ij,ik,il->jkl"(ca, ca, ca)) - ein"ij,ik,il->jkl"(a, a, a))) < 1e-4
end

b = randn(Float32, 100, 50)
x1 = randn(100, 50)
x2 = randn(50, 50)
locs_xs = ((1,2), (2,3))
ixs = ((1,2), (2,3))
iy = (1,3)
a = EinArray(EinCode(ixs, iy), (x1, x2), OMEinsum.get_size_dict(ixs, (x1, x2)))
@test size(a) == (50,100,50) # inner, outer
ca = cu(a);
out = zeros(1,100, 50) |> cu

#CUDAnative.isghosttype(::Type{T}) where T<:EinArray = true
#Base.isbitstype(::Type{T}) where T<:Base.OneTo = true
Matrix(dropdims(Base._mapreducedim!(x->x, +, out, ca), dims=1)) ≈ dropdims(sum(Array(a), dims=1), dims=1)
