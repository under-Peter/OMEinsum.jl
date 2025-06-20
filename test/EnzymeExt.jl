using Enzyme, OMEinsum, Test
function testf1(x)
    y = zeros(size(x, 1))
    einsum!(ein"ii->i", (x,), y, 1, 0, Dict('i'=>3))
    return sum(y)
end

function testf2(x)
    y = einsum(ein"ii->i", (x,), Dict('i'=>3))
    return sum(y)
end

function testf4(x)
    y = ein"ii->i"(x)
    return sum(y)
end

@testset "EnzymeExt" begin
    x = randn(3, 3);
    gx = zero(x);

    autodiff(ReverseWithPrimal, testf1, Active, Duplicated(x, gx))
    @test gx == [1 0 0; 0 1 0; 0 0 1]

    autodiff(ReverseWithPrimal, testf2, Active, Duplicated(x, gx))
    @test gx == [2 0 0; 0 2 0; 0 0 2]

    autodiff(ReverseWithPrimal, testf4, Active, Duplicated(x, gx))
    @test gx == [3 0 0; 0 3 0; 0 0 3]
end

@testset "EnzymeExt error" begin
    x = randn(3, 3);
    gx = zero(x);
    function testf3(x)
        y = zeros(size(x, 1))
        einsum!(ein"ii->i", (x,), y, 1, 0, Dict('i'=>3))
        return sum(y)
    end
    autodiff(ReverseWithPrimal, testf3, Active, Duplicated(x, gx))
    @test gx == [1 0 0; 0 1 0; 0 0 1]
end

@testset "EnzymeExt bp check" begin
    A, B, C = randn(2, 3), randn(3, 4), randn(4, 2)
    cost0 = ein"(ij, jk), ki->"(A, B, C)[]
    gA = zero(A); gB = zero(B); gC = zero(C);
    Enzyme.autodiff(Reverse, (a, b, c)->ein"(ij, jk), ki->"(a, b, c)[], Active, Duplicated(A, gA), Duplicated(B, gB), Duplicated(C, gC))
    cost, mg = OMEinsum.cost_and_gradient(ein"(ij, jk), ki->", (A, B, C))
    @test cost[] ≈ cost0
    @test all(gA .≈ mg[1])
    @test all(gB .≈ mg[2])
    @test all(gC .≈ mg[3])
end

@testset "EnzymeExt bp check 2" begin
    A, B, C = randn(2, 3), randn(3, 4), randn(4, 2)
    code = optimize_code(ein"ij, jk, ki->", uniformsize(ein"ij, jk, ki->", 2), TreeSA())
    cost0 = code(A, B, C)[]
    gA = zero(A); gB = zero(B); gC = zero(C);
    f(code, a, b, c) = code(a, b, c)[]
    Enzyme.autodiff(set_runtime_activity(Reverse), f, Active, Const(code), Duplicated(A, gA), Duplicated(B, gB), Duplicated(C, gC))
    cost, mg = OMEinsum.cost_and_gradient(code, (A, B, C))
    @test cost[] ≈ cost0
    @test all(gA .≈ mg[1])
    @test all(gB .≈ mg[2])
    @test all(gC .≈ mg[3])
end

# liquid state machine