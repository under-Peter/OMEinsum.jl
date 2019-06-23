module OMEinsum
export einsum, expandall!
export einsumopt

using TupleTools, Requires, TensorOperations, LinearAlgebra

asarray(x::Number) = fill(x, ())
asarray(x::AbstractArray) = x

include("einsumexp.jl")
include("einsum.jl")
include("einsumopt.jl")
#include("einorder.jl")
#include("einevaluate.jl")
@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cueinsum.jl")
include("interfaces.jl")
include("autodiff.jl")
end # module
