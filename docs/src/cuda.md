# CUDA Acceleration

By uploading your data to the GPU, you can accelerate the computation of your model.

```julia repl
julia> using CUDA, OMEinsum

julia> code = ein"ij,jk,kl,li->"  # the einsum notation
ij, jk, kl, li -> 

julia> A, B, C, D = rand(1000, 1000), rand(1000, 300), rand(300, 800), rand(800, 1000);

julia> size_dict = OMEinsum.get_size_dict(getixsv(code), (A, B, C, D))  # get the size of the labels
Dict{Char, Int64} with 4 entries:
  'j' => 1000
  'i' => 1000
  'k' => 300
  'l' => 800

julia> optcode = optimize_code(code, size_dict, TreeSA())  # optimize the contraction order
SlicedEinsum{Char, DynamicNestedEinsum{Char}}(Char[], kl, kl -> 
├─ ki, li -> kl
│  ├─ jk, ij -> ki
│  │  ├─ jk
│  │  └─ ij
│  └─ li
└─ kl
)
```

The contraction order is optimized. Now, let's benchmark the contraction on the CPU.

```julia repl
julia> using BenchmarkTools

julia> @btime optcode($A, $B, $C, $D)  # the contraction on CPU
  6.053 ms (308 allocations: 20.16 MiB)
0-dimensional Array{Float64, 0}:
1.4984046443610943e10
```

The contraction on the CPU takes about 6 ms. Now, let's upload the data to the GPU and perform the contraction on the GPU.
```julia repl
julia> @btime CUDA.@sync optcode($cuA, $cuB, $cuC, $cuD)  # the contraction on GPU
  243.888 μs (763 allocations: 28.56 KiB)
0-dimensional CuArray{Float64, 0, CUDA.DeviceMemory}:
1.4984046443610939e10
```