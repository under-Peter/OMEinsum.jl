using OMEinsum
using Test
using LinearAlgebra

@testset "OMEinsum.jl" begin
    include("einsum.jl")
    include("autodiff.jl")
end
