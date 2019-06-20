using OMEinsum
using Test
using LinearAlgebra

@testset "OMEinsum.jl" begin
    include("einorder.jl")
    include("einsum.jl")
    include("autodiff.jl")
end
