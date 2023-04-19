# ConfigWidth{1} means our rule does not support batched mode
using EnzymeCore: Active, Duplicated, Const, EnzymeRules

function EnzymeRules.augmented_primal(
                config::EnzymeRules.ConfigWidth{1},
                func::Const{typeof(einsum)}, ::Type{<:Duplicated}, 
                code::Const, xs::Duplicated, size_dict)
    @debug "In custom augmented primal rule: $code"
    # Compute primal
    if EnzymeRules.needs_primal(config)
        primal = func.val(code.val, xs.val, size_dict.val); 
                shadow=zero(primal)
    else
        primal, shadow = nothing, nothing
    end
    # Save x in tape if x will be overwritten
    if EnzymeRules.overwritten(config)[3]
        tape = [copy.(xs.val)..., shadow]
    else
        tape = [shadow]
    end
    @debug "tape = $tape"
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(config::EnzymeRules.ConfigWidth{1},
               func::Const{typeof(einsum)}, dret::Type{<:Duplicated}, tape,
               code::Const,
               xs::Duplicated, size_dict)
    @debug "In custom reverse rule: $config, got tape: $tape"
    dval = tape[end]
    if EnzymeRules.overwritten(config)[3]
        xval = tape[1:end-1]
    else
        xval = xs.val
    end
    for i=1:length(xs.val)
        xs.dval[i] .+= OMEinsum.einsum_grad(OMEinsum.getixs(code.val),
             xval, OMEinsum.getiy(code.val), size_dict.val, conj(dval), i)
    end
    return ()
end