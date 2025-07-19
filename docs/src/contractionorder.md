# Contraction order optimization

## Constructing a code
The [`@ein_str`](@ref) string literal does not optimize the contraction order for more than two input tensors.
We first use a graph to construct a [`DynamicEinCode`](@ref) object for demonstration.

```@repl order
using OMEinsum, OMEinsumContractionOrders, OMEinsumContractionOrders.Graphs

graph = random_regular_graph(20, 3; seed=42)

code = EinCode([[e.src, e.dst] for e in edges(graph)], Int[])
```

The return value is a [`StaticEinCode`](@ref) object that does not contain a contraction order.
The time and space complexity can be obtained by calling the [`contraction_complexity`](@ref) function.
```@repl order
size_dict = uniformsize(code, 2)  # size of the labels are set to 2

contraction_complexity(code, size_dict)  # time and space complexity
```

The return values are `log2` values of the number of iterations, number of elements of the largest tensor and the number of elementwise read-write operations.

## Optimizing the contraction order
To optimize the contraction order, we can use the [`optimize_code`](@ref) function.

```@repl order
optcode = optimize_code(code, size_dict, TreeSA(ntrials=1))
```

The output value is a binary contraction tree with type [`SlicedEinsum`](@ref) or [`NestedEinsum`](@ref).
The `TreeSA` is a local search algorithm that optimizes the contraction order. More algorithms can be found in the
[OMEinsumContractionOrders documentation](https://tensorbfs.github.io/OMEinsumContractionOrders.jl/dev/).

After optimizing the contraction order, the time and readwrite complexities are significantly reduced.

```@repl order
contraction_complexity(optcode, size_dict)
```

## Slicing the code
In some cases, the memory usage of the contraction is too large.
Slicing is a technique to reduce the time and space complexity of the contraction.
The slicing is done by using the [`slice_code`](@ref) function.
```@repl order
slicer = TreeSASlicer(score=ScoreFunction(sc_target=2))
scode = slice_code(optcode, size_dict, slicer);
contraction_complexity(scode, size_dict)
scode.slicing
```
The return value is a [`SlicedEinsum`](@ref) object. The space complexity is reduced to 2, while the time complexity is increased as a trade-off.

## Using `optein` string literal
For convenience, the optimized contraction can be directly contructed by using the [`@optein_str`](@ref) string literal.
```@repl order
optein"ij,jk,kl,li->"  # optimized contraction, without knowing the size of the tensors
```
`@optein_str` optimizes the contraction order with the assumption that each index has the same size 2, hence the resulting contraction order might not be optimal.

## Manual optimization
One can also manually specify the contraction order by using the [`@ein_str`](@ref) string literal.
```@repl order
ein"((ij,jk),kl),li->ik"  # manually optimized contraction
```

## Flatten the code

Given an optimized code, one can flatten it to get a code without contraction order with type [`EinCode`](@ref).

```@repl order
OMEinsum.flatten(optcode)
```