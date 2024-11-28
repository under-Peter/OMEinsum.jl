module EnzymeExt
using Enzyme.EnzymeRules, OMEinsum, Enzyme
using OMEinsum: get_size_dict!

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(einsum!)}, ::Type, 
        code::Const, xs::Duplicated, ys::Duplicated, sx::Const, sy::Const, size_dict::Const)
    @assert sx.val == 1 && sy.val == 0 "Only α = 1 and β = 0 is supported, got: $sx, $sy"
    # Compute primal
    if EnzymeRules.needs_primal(config)
        primal = func.val(code.val, xs.val, ys.val, sx.val, sy.val, size_dict.val)
    else
        primal = nothing
    end
    # Save x in tape if x will be overwritten
    if EnzymeRules.overwritten(config)[3]
        tape = copy(xs.val)
    else
        tape = nothing
    end
    shadow = ys.dval
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
               func::Const{typeof(einsum!)}, dret::Type{<:Annotation}, tape,
               code::Const,
               xs::Duplicated, ys::Duplicated, sx::Const, sy::Const, size_dict::Const)
   xval = EnzymeRules.overwritten(config)[3] ? tape : xs.val
   for i=1:length(xs.val)
       xs.dval[i] .+= OMEinsum.einsum_grad(OMEinsum.getixs(code.val),
             xval, OMEinsum.getiy(code.val), size_dict.val, conj(ys.dval), i)
   end
   return (nothing, nothing, nothing, nothing, nothing, nothing)
end

# EnzymeRules.inactive(::typeof(get_size_dict!), args...) = nothing

end