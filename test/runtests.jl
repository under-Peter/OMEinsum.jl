using OMEinsum
using Test
using LinearAlgebra


@testset "OMEinsum.jl" begin
    include("Core.jl")
    include("EinRule.jl")
    include("utils.jl")
    include("einsum.jl")
    if Base.find_package("CuArrays") != nothing
        include("cueinsum.jl")
    end
    include("autodiff.jl")
    include("einsequence.jl")
    include("interfaces.jl")
end
