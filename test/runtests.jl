using OMEinsum
using Test
using LinearAlgebra

@testset "OMEinsum.jl" begin
    include("EinsumOp.jl")
    include("einsum.jl")
    include("autodiff.jl")
    #include("einorder.jl")
    #include("einsumopt.jl")
end
