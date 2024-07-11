using Documenter, OMEinsum

makedocs(;
    modules=[OMEinsum],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Background: Tensor Networks" => "background.md",
        "Basic usage" => "basic.md",
        "Contraction order optimization" => "contractionorder.md",
        "Automatic differentiation" => "autodiff.md",
        "CUDA" => "cuda.md",
        "Manual" => "docstrings.md"
    ],
    repo="https://github.com/under-Peter/OMEinsum.jl/blob/{commit}{path}#L{line}",
    sitename="OMEinsum.jl",
    authors="Andreas Peter",
)

deploydocs(;
    repo="github.com/under-Peter/OMEinsum.jl",
)
