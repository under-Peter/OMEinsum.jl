using Test
using OMEinsum
using OMEinsum: StaticEinCode, DynamicEinCode
using Zygote

@doc raw"
    bpcheck(f, args...; η = 1e-5, verbose=false)
returns a `Bool` indicating whether Zygote calculates the gradient of `f(args...) -> scalar`
correctly using the relation `f(x - ηg) ≈ f(x) - η|g|²` with a relative tolerance
of 1e-2 and an absolute tolerance of 1e-8.
If `verbose=true`, print `f(x) - f(x - ηg)`and `η|g|²`.

# example

```jldoctest; setup = :(using OMEinsum)
julia> using OMEinsum: bpcheck

julia> a, b = rand(2,2), rand(2,2);

julia> bpcheck(sum ∘ ein\"ij,jk -> ik\", a, b)
true
```
"
function bpcheck(f, args...; η = 1e-5, verbose = false)
    g = gradient(f, args...)
    all(==(nothing), g) && error()
    dy_ref = 0
    for x in g
        x === nothing && continue
        x isa Tuple && (dy_ref += η * mapreduce(y -> y == nothing ? 0 : sum(abs2,y), +, x))
        x isa AbstractArray && (dy_ref += η * sum(abs2,x))
    end
    dy = f(args...) - f([gi === nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args,g)]...)

    verbose && @show dy
    verbose && @show dy_ref

    isapprox(dy, dy_ref, rtol=1e-2, atol=1e-8)
end


@testset "einsum bp" begin
    for T in (Float64, ComplexF64)
        @testset "$T" begin
            # matrix and vector multiplication
            a,b,c = rand(T,2,2), rand(T,2,2), rand(T,2,2)
            v = rand(T,2)
            t = randn(2,2,2,2)
            @test bpcheck( (a,b,c) -> einsum(EinCode(((1,2),(2,3),(3,4)), (1,4)), (a,b,c)) |> abs ∘ sum ,a,b,c)
            @test bpcheck( (a,b,c) -> einsum(EinCode(((1,2),(2,3),(3,4)), (4,1)), (a,b,c)) |> abs ∘ sum ,a,b,c)
            @test bpcheck((a,v) -> einsum(EinCode(((1,2),(2,)), (1,)), (a,v)) |> abs ∘ sum , a, v)
            @test bpcheck((a,v) -> einsum(StaticEinCode{((1,2),(2,)), (1,)}(), (a,v)) |> abs ∘ sum , a, v)
            @test bpcheck((a,t,v) -> ein"(ab,abcd),c->ad"(a,t,v) |> abs ∘ sum ,a,T.(t),v)

            # contract to 0-dim array
            @test bpcheck((a,b) -> einsum(EinCode(((1,2),(1,2)), ()), (a,b)) |> abs ∘ sum , a,b)

            # trace
            @test bpcheck(a -> einsum(EinCode(((1,1),), ()), (a,)) |> abs ∘ sum, a)
            aa = rand(T,2,4,4,2)
            @test bpcheck(aa -> einsum(EinCode(((1,2,2,1),), ()), (aa,)) |> abs ∘ sum, aa)

            # partial trace
            @test bpcheck(aa -> einsum(EinCode(((1,2,2,3),), (1,3)), (aa,)) |> abs ∘ sum, aa)

            # diag
            @test bpcheck(aa -> einsum(EinCode(((1,2,2,3),), (1,2,3)), (aa,)) |> abs ∘ sum, aa)

            # permutation
            @test bpcheck(a -> einsum(EinCode(((1,2),), (2,1)), (a,)) |> abs ∘ sum, a)
            @test bpcheck(t -> einsum(EinCode(((1,2,3,4),),(2,3,1,4)), (t,)) |> abs ∘ sum, t)

            # tensor contraction
            @test bpcheck((t,a) -> einsum(EinCode(((1,2,3,4), (2,3)), (1,4)), (t,a)) |> abs ∘ sum, t,a)
            @test bpcheck((t,a) -> einsum(EinCode(((4,3,2,1), (2,3)), (1,4)), (t,a)) |> abs ∘ sum, t,a)

            # star-contraction
            @test bpcheck((a,b,c) -> einsum(EinCode(((1,2),(1,3),(1,4)), (2,3,4)), (a,b,c)) |> abs ∘ sum, a,b,c)

            # star and contract
            @test bpcheck((a,b,c) -> einsum(EinCode(((1,2),(1,2),(1,3)), (3,)), (a,b,c)) |> abs ∘ sum, a,b,c)

            # index-sum
            a3 = rand(T,2,2,2)
            @test bpcheck(a -> einsum(EinCode(((1,2,3),),(1,2)),(a,)) |> abs ∘ sum, a3)

            # Hadamard product
            @test bpcheck((a,b) -> einsum(EinCode(((1,2),(1,2)), (1,2)), (a,b)) |> abs ∘ sum, a, b)

            # Outer
            @test bpcheck((a,b) -> einsum(EinCode(((1,2),(3,4)),(1,2,3,4)),(a,b)) |> abs ∘ sum, a, b)
        end
    end
end

@testset "gradient type check" begin
    array_match(x, y) = typeof(x) == typeof(y) && size(x) == size(y)
    a = randn(3,3)
    b = randn(3,3)
    @test array_match(gradient(a->einsum(EinCode(((1,2), (2,1)), ()), (a, b))[] |> abs, a)[1], a)
    b = randn(ComplexF64,3,3)
    @test array_match(gradient(a->einsum(EinCode(((1,2), (2,1)), ()), (a, b))[] |> abs, a)[1], a)
    a = randn(ComplexF64,3,3)
    @test array_match(gradient(a->einsum(EinCode(((1,2), (2,3)), ()), (a, b))[] |> abs, a)[1], a)
    b = randn(3,3)
    @test array_match(gradient(a->einsum(EinCode(((1,2), (2,3)), ()), (a, b))[] |> abs, a)[1], a)
end

@testset "string-specification" begin
    a,b,c = rand(2,2), rand(2,2), rand(2,2)
    v = rand(2)
    @test bpcheck((a,b,c) -> einsum(ein"ij,jk,kl -> il", (a,b,c)) |> abs ∘ sum ,a,b,c)
    @test bpcheck((a,b,c) -> einsum(ein"ij,jk,kl -> li", (a,b,c)) |> abs ∘ sum ,a,b,c)
    @test bpcheck((a,v) -> einsum(ein"ij,j -> i", (a,v)) |> abs ∘ sum , a, v)
end

@testset "sequence specification" begin
    a, b, c = rand(2,2), rand(2,2), rand(2,2)
    @test all(gradient(sum ∘ ein"ij,jk,kl -> il", a, b, c) .≈
          gradient(sum ∘ ein"(ij,(jk,kl)) -> il", a, b, c))
end
