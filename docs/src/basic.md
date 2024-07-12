# Basic Usage

In the following example, we demonstrate the einsum notation for basic tensor operations.

## Einsum notation
To specify the operation, the user can either use the [`@ein_str`](@ref)-string literal or the [`EinCode`](@ref) object.
For example, both the following code snippets define the matrix multiplication operation:
```@repl tensor
using OMEinsum
code1 = ein"ij,jk -> ik"  # the string literal
ixs = [[1, 2], [2, 3]]  # the input indices
iy = [1, 3]  # the output indices
code2 = EinCode(ixs, iy)  # the EinCode object (equivalent to the string literal)
```

The [`@ein_str`](@ref) macro can be used to define the einsum notation directly in the function call.
```@repl tensor
A, B = randn(2, 3), randn(3, 4);
code1(A, B)  # matrix multiplication
size_dict = OMEinsum.get_size_dict(getixsv(code1), (A, B))  # get the size of the labels
einsum(code1, (A, B), size_dict)  # lower-level function
einsum!(code1, (A, B), zeros(2, 4), true, false, size_dict)  # the in-place operation
@ein C[i,k] := A[i,j] * B[j,k]  # all-in-one macro
```
Here, we show that the [`@ein`](@ref) macro combines the einsum notation defintion and the operation in a single line, which is more convenient for simple operations.
Separating the einsum notation and the operation (the first approach) can be useful for reusing the einsum notation for multiple input tensors.
Lower level functions, [`einsum`](@ref) and [`einsum!`](@ref), can be used for more control over the operation.

For more than two input tensors, *the [`@ein_str`](@ref) macro does not optimize the contraction order*. In such cases, the user can use the [`@optein_str`](@ref) string literal to optimize the contraction order or specify the contraction order manually.
```@repl tensor
tensors = [randn(100, 100) for _ in 1:4];
optein"ij,jk,kl,lm->im"(tensors...)  # optimized contraction (without knowing the size)
ein"(ij,jk),(kl,lm)->im"(tensors...)  # manually specified contraction
```

Sometimes, manually optimizing the contraction order can be beneficial. Please check [Contraction order optimization](@ref) for more details.

## Einsum examples
We first define the tensors and then demonstrate the einsum notation for various tensor operations.
```@repl tensor
using OMEinsum
s = fill(1)  # scalar
w, v = [1, 2], [4, 5];  # vectors
A, B = [1 2; 3 4], [5 6; 7 8]; # matrices
T1, T2 = reshape(1:8, 2, 2, 2), reshape(9:16, 2, 2, 2); # 3D tensor
```
### Unary examples
```@repl tensor
ein"i->"(w)  # sum of the elements of a vector.
ein"ij->i"(A)  # sum of the rows of a matrix.
ein"ii->"(A)  # sum of the diagonal elements of a matrix, i.e., the trace.
ein"ij->"(A)  # sum of the elements of a matrix.
ein"i->ii"(w)  # create a diagonal matrix.
ein"i->ij"(w; size_info=Dict('j'=>2))  # repeat a vector to form a matrix.
ein"ijk->ikj"(T1)  # permute the dimensions of a tensor.
```

### Binary examples
```@repl tensor
ein"ij, jk -> ik"(A, B)  # matrix multiplication.
ein"ijb,jkb->ikb"(T1, T2)  # batch matrix multiplication.
ein"ij,ij->ij"(A, B)  # element-wise multiplication.
ein"ij,ij->"(A, B)  # sum of the element-wise multiplication.
ein"ij,->ij"(A, s)  # element-wise multiplication by a scalar.
```

### Nary examples
```@repl tensor
optein"ai,aj,ak->ijk"(A, A, B)  # star contraction.
optein"ia,ajb,bkc,cld,dm->ijklm"(A, T1, T2, T1, A)  # tensor train contraction.
```