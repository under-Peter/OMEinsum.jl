# Contraction order optimization

OMEinsum does not implicitly optimize the contraction order.
Functionalities related to contraction order optimization are mostly defined in [OMEinsumContractionOrders](https://github.com/TensorBFS/OMEinsumContractionOrders.jl)

Here, we provide an example, advanced uses can be found in [OMEinsumContractionOrders](https://github.com/TensorBFS/OMEinsumContractionOrders.jl) and the [performance tips](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/performancetips/) of [GenericTensorNetworks](https://github.com/QuEraComputing/GenericTensorNetworks.jl).
Let us first consider the following contraction order

```@example 3
using OMEinsum, OMEinsumContractionOrders

code = ein"ij,jk,kl,li->"
```

The time and space complexity can be obtained by calling the [`timespacereadwrite_complexity`](@ref) function.
```@example 3
size_dict = uniformsize(code, 10)

timespacereadwrite_complexity(code, size_dict)
```

The return values are `log2` values of the number of iterations, number of elements of the largest tensor and the number of elementwise read-write operations.

```@example 3
optcode = optimize_code(code, size_dict, TreeSA())
```

The output value is a binary contraction tree with type [`NestedEinsum`](@ref) type.
The time and readwrite complexities are significantly reduced comparing to the direct contraction.

```@example 3
timespacereadwrite_complexity(optcode, size_dict)
```