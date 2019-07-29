using ProgressMeter
pmobj = Progress(205, )
using OMEinsum
using Test
using LinearAlgebra


@testset "OMEinsum.jl" begin
    include("EinRule.jl")
    include("utils.jl")
    include("einsum.jl")
    include("autodiff.jl")
    include("einorder.jl")
    include("einsumopt.jl")
    include("einsequence.jl")
end
println()