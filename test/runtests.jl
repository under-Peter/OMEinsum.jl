using OMEinsum
using Test
using LinearAlgebra

@testset "OMEinsum.jl" begin
    include("einsum.jl")
    include("expanding.jl")
end
