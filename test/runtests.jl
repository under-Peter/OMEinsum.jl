using OMEinsum
using Test

@testset "autodiff" begin
    include("einsum.jl")
end

@testset "autodiff" begin
    include("autodiff.jl")
end
