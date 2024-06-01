using OMEinsum
using OMEinsum: cost_and_gradient
A, B, C = randn(2, 3), randn(3, 4), randn(4, 2)
y, g = cost_and_gradient(ein"(ij, jk), ki->", (A, B, C))

using Zygote
xA, xB, xC = randn(2, 3), randn(3, 4), randn(4, 2)
ȳ = fill(1.0)
function gfunc(A, B, C)
    cost, (gA, gB, gC) = cost_and_gradient(ein"(ij, jk), ki->", (A, B, C), ȳ)
    return sum(gA .* xA) + sum(gB .* xB) + sum(gC .* xC)
end
Zygote.gradient(gfunc, A, B, C)

using ReverseDiff
ReverseDiff.@grad_from_chainrules einsum(args...; kwargs...)

ReverseDiff.gradient(gfunc, (A, B, C))