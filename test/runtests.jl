using OMEinsum
using Test
using LinearAlgebra


@testset "OMEinsum.jl" begin
    include("EinRule.jl")
    include("utils.jl")
    include("einsum.jl")
    if Base.find_package("CuArrays") != nothing
        include("cueinsum.jl")
    end
    include("autodiff.jl")
    include("einorder.jl")
    include("einsumopt.jl")
    include("einsequence.jl")
end
