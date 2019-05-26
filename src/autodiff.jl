using Zygote
using Zygote: @adjoint!

export bpcheck

# TODO show which is faster
tuple_replace(t::Tuple, i, x) = (t[1:i-1]..., x, t[i+1:end]...)
#tuple_replace(t::Tuple, i, x) = map(k->k==i ? x : t[k], Tuple(1:Ni))

function eingrad_i(ixs, xs, iy, cdy, i::Int)
    Ni = length(ixs)
    adjy = einmagic!(tuple_replace(ixs, i, iy), tuple_replace(xs, i, cdy), ixs[i], zero(xs[i]))
    return adjy |> conj
end

@adjoint! function einmagic!(ixs::Tuple, xs, iy, y)
    einmagic!(ixs, xs, iy, y),
    function(dy)
        @show dy
        cdy = conj(dy)
        return (nothing, Tuple(eingrad_i(ixs, xs, iy, cdy, i::Int) for i=1:length(xs)), nothing, nothing)
    end
end

##### Gradient Check
superabs2(x) = mapreduce(superabs2, +, x)
superabs2(x::Number) = abs2(x)

"""
Give g the gradients, α a small number, for a analytical function, we have

    f(x-αg) ≈ f(x) - α|g|²
"""
function bpcheck(f, args...; η = 1e-5, showy=false)
    g = gradient(f, args...)
    dy_expect = η*superabs2(g)
    dy = f(args...)-f([gi == nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    if showy
        @show dy
        @show dy_expect
    end
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end
