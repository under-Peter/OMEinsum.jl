using OMEinsum
using Test
using LinearAlgebra
using CUDA
import Documenter


@testset "Core" begin
    include("Core.jl")
end

@testset "match rule" begin
    include("matchrule.jl")
end

@testset "unary rules" begin
    include("unaryrules.jl")
end

@testset "binary rules" begin
    include("binaryrules.jl")
end

@testset "utils" begin
    include("utils.jl")
end

@testset "einsum" begin
    include("einsum.jl")
end

@testset "cuda" begin
    if CUDA.functional()
        include("cueinsum.jl")
    end
end

@testset "autodiff" begin
    include("autodiff.jl")
end

@testset "einsequence" begin
    include("einsequence.jl")
end

@testset "slicing" begin
    include("slicing.jl")
end

@testset "interfaces" begin
    include("interfaces.jl")
end

@testset "contraction order" begin
    include("contractionorder.jl")
end

@testset "back propagation" begin
    include("bp.jl")
end

@testset "EnzymeExt" begin
    include("EnzymeExt.jl")
end

@testset "docstring" begin
    Documenter.doctest(OMEinsum; manual=false)
end

@testset "deprecation" begin
    code = EinCode([[1,2], [2,3], [3,4]], [1,4])
    @test_deprecated optimize_code(code, uniformsize(code, 2), TreeSA(ntrials=1), MergeVectors())
end