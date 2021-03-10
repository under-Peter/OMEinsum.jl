using OMEinsum
using Test
using LinearAlgebra
using CUDA


@testset "OMEinsum.jl" begin
    include("Core.jl")
    include("EinRule.jl")
    include("unaryrules.jl")
    include("binaryrules.jl")
    include("utils.jl")
    include("einsum.jl")
    if CUDA.functional()
        include("cueinsum.jl")
    end
    include("autodiff.jl")
    include("einsequence.jl")
    include("interfaces.jl")
end
