# Automatic differentiation

There are two ways to compute the gradient of an einsum expression. The first one is to use the `OMEinsum` package, which is a custom implementation of the reverse-mode automatic differentiation. The second one is to use the [`Zygote`](https://github.com/FluxML/Zygote.jl) package, which is a source-to-source automatic differentiation tool.

## Built-in automatic differentiation
The `OMEinsum` package provides a built-in function [`cost_and_gradient`](@ref) to compute the cost and the gradient of an einsum expression.

```@repl autodiff
using OMEinsum  # the 1st way
A, B, C = randn(2, 3), randn(3, 4), randn(4, 2);
y, g = cost_and_gradient(ein"(ij, jk), ki->", (A, B, C))
```
This built-in automatic differentiation is designed for tensor contractions and is more efficient than the general-purpose automatic differentiation tools.

For complex valued tensors, the automatic differentiation is defined in a convention that treat the real and imaginary parts as independent variables.

## Using Zygote
The backward rule for the basic einsum operation is ported to the [`ChainRulesCore`](https://github.com/JuliaDiff/ChainRulesCore.jl), which is used by the `Zygote` package.
Zygote is a source-to-source automatic differentiation tool that can be used to compute the gradient of an einsum expression.
It is more general and can be used for any Julia code.
```@repl autodiff
using Zygote  # the 2nd way
Zygote.gradient((A, B, C)->ein"(ij, jk), ki->"(A, B, C)[], A, B, C)
```