using OMEinsum
using Test
using LinearAlgebra
using CUDA
import Documenter


@testset "OMEinsum.jl" begin
    include("Core.jl")
    include("EinRule.jl")
    include("binaryrules.jl")
    include("utils.jl")
    include("einsum.jl")
    if CUDA.functional()
        include("cueinsum.jl")
    end
    include("autodiff.jl")
    include("einsequence.jl")
    include("interfaces.jl")

    include("contractionorder.jl")

    Documenter.doctest(OMEinsum; manual=false)
end
