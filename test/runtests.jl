using OMEinsum
using Test
using LinearAlgebra

@testset "OMEinsum.jl" begin
    include("utils.jl")
    include("einsum.jl")
    include("autodiff.jl")
    include("einorder.jl")
end
