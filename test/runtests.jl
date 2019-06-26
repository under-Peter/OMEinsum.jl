using OMEinsum
using Test
using LinearAlgebra

@testset "EinRule" begin
    include("EinRule.jl")
end

@testset "utils.jl" begin
    include("utils.jl")
end

@testset "einsum" begin
    include("einsum.jl")
end

@testset "autodiff" begin
    include("autodiff.jl")
end

@testset "einorder" begin
    include("einorder.jl")
end

@testset "einsumopt" begin
    include("einsumopt.jl")
end
