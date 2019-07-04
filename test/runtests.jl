using OMEinsum
using Test
using LinearAlgebra
using ProgressMeter

p = ProgressUnknown("Testset-running:")


@testset "OMEinsum.jl" begin
    next!(p)
    include("EinRule.jl")
    next!(p)
    include("utils.jl")
    next!(p)
    include("einsum.jl")
    next!(p)
    include("autodiff.jl")
    next!(p)
    include("einorder.jl")
    next!(p)
    include("einsumopt.jl")
    finish!(p)
end
