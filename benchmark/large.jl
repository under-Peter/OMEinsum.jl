using OMEinsum, BenchmarkTools

function benchmark_tensorpermute()
    # tensorpermute
    A = randn(fill(2, 28)...);
    C = zero(A);
    perm = [18  22  11  21  15  9  10  19  24  14  5  1  17  20 26 25 28 27 7  6  3  13  12  16  8  23  2  4] |> vec
    @btime OMEinsum.tensorpermute!($C, $A, $perm, true, false) evals=3
    nothing
end