using Zygote
using Zygote: @adjoint

export bpcheck

einsum!(contractions, tensors, outinds, outtensor) = copyto!(outtensor, einsum(contractions, tensors, outinds))

function eingrad_i(ixs, xs, iy, cdy, i::Int)
    Ni = length(ixs)
    einsum!(map(k->k==i ? iy : ixs[k], 1:Ni), map(k->k==i ? cdy : xs[k], 1:Ni), ixs[i], zero(xs[i])) |> conj
end

@adjoint function einsum!(ixs, xs, iy, y)
    einsum!(ixs, xs, iy, y)
    y, dy -> (cdy=conj(dy); (nothing, Tuple(eingrad_i(ixs, xs, iy, cdy, i::Int) for i=1:length(xs)), nothing, nothing))
end

"""
Give g the gradients, α a small number, for a analytical function, we have

    f(x-αg) ≈ f(x) - α|g|²
"""
function bpcheck(f, args...; η = 1e-5, showy=false)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi == nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    if showy
        @show dy
        @show dy_expect
    end
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end
