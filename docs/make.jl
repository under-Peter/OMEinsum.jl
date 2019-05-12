using Documenter, OMEinsum

makedocs(;
    modules=[OMEinsum],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/under-Peter/OMEinsum.jl/blob/{commit}{path}#L{line}",
    sitename="OMEinsum.jl",
    authors="Andreas Peter",
)

deploydocs(;
    repo="github.com/under-Peter/OMEinsum.jl",
)
