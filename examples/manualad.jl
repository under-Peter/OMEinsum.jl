using OMEinsum
using OMEinsum: cost_and_gradient
A, B, C = randn(2, 3), randn(3, 4), randn(4, 2)
y, g = cost_and_gradient(ein"(ij, jk), ki->", (A, B, C))

# evaluate the cost and the gradient of leaves
function gf(code, xs, res, ȳ = OMEinsum.init_gradient(code, xs))
    cost, tree = OMEinsum.gradient_tree(code, xs, ȳ)
    # extract the gradients on leaves (i.e. the input tensors).
    return cost, OMEinsum.extract_leaves!(code, tree, res)
end

using Zygote
xA, xB, xC = randn(2, 3), randn(3, 4), randn(4, 2)
function gfunc(A, B, C)
    ȳ = fill(one(eltype(A)))
    res = Zygote.Buffer(Any[nothing, nothing, nothing])
    cost, (gA, gB, gC) = gf(ein"(ij, jk), ki->", (A, B, C), res, ȳ)
    return sum(gA .* xA) + sum(gB .* xB) + sum(gC .* xC)
end

g = Zygote.gradient(gfunc, A, B, C)