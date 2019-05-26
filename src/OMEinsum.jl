module OMEinsum
using TupleTools
export einsum, expandall!

include("einsum.jl")
include("EinCode.jl")
include("autodiff.jl")

end # module
