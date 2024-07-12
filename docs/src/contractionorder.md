# Contraction order optimization

The [`@ein_str`](@ref) string literal does not optimize the contraction order for more than two input tensors.

```@repl order
using OMEinsum

code = ein"ij,jk,kl,li->"
```

The return value is a [`StaticEinCode`](@ref) object that does not contain a contraction order.
The time and space complexity can be obtained by calling the [`contraction_complexity`](@ref) function.
```@repl order
size_dict = uniformsize(code, 10)  # size of the labels are set to 10

contraction_complexity(code, size_dict)  # time and space complexity
```

The return values are `log2` values of the number of iterations, number of elements of the largest tensor and the number of elementwise read-write operations.

## Optimizing the contraction order
To optimize the contraction order, we can use the [`optimize_code`](@ref) function.

```@repl order
optcode = optimize_code(code, size_dict, TreeSA())
```

The output value is a binary contraction tree with type [`SlicedEinsum`](@ref) or [`NestedEinsum`](@ref).
The `TreeSA` is a local search algorithm that optimizes the contraction order. More algorithms can be found in the
[OMEinsumContractionOrders](https://github.com/TensorBFS/OMEinsumContractionOrders.jl) and the [performance tips](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/performancetips/) of [GenericTensorNetworks](https://github.com/QuEraComputing/GenericTensorNetworks.jl).

After optimizing the contraction order, the time and readwrite complexities are significantly reduced.

```@repl order
contraction_complexity(optcode, size_dict)
```

## Using `optein` string literal
For convenience, the optimized contraction can be directly contructed by using the [`@optein_str`](@ref) string literal.
```@repl order
optein"ij,jk,kl,li->"  # optimized contraction, without knowing the size of the tensors
```
The drawback of using `@optein_str` is that the contraction order is optimized without knowing the size of the tensors.
Only the tensor ranks are used to optimize the contraction order.

## Manual optimization
One can also manually specify the contraction order by using the [`@ein_str`](@ref) string literal.
```@repl order
ein"((ij,jk),kl),li->ik"  # manually optimized contraction
```