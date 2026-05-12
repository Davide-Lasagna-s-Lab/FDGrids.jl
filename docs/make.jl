using Documenter
using FDGrids
using LinearAlgebra

DocMeta.setdocmeta!(
    FDGrids,
    :DocTestSetup,
    :(using FDGrids, LinearAlgebra, Random);
    recursive = true,
)

makedocs(
    sitename = "FDGrids.jl",
    modules = [FDGrids],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://Davide-Lasagna-s-Lab.github.io/FDGrids.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting-started.md",
            "Dimension-Wise Differentiation" => "tutorials/dimension-wise.md",
            "Weighted Adjoints" => "tutorials/weighted-adjoints.md",
            "Decomposed Domains" => "tutorials/decomposed-domains.md",
        ],
        "Manual" => [
            "Grids and Quadrature" => "manual/grids.md",
            "Finite-Difference Operators" => "manual/diffmatrix.md",
            "Adjoints" => "manual/adjoints.md",
            "Linear Solves" => "manual/linear-solves.md",
            "Numerical Methods" => "manual/methods.md",
            "Internal Layout and Kernels" => "manual/internals.md",
        ],
        "API Reference" => "api.md",
        "Deprecated API" => "deprecated.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/Davide-Lasagna-s-Lab/FDGrids.jl.git",
    devbranch = "master",
)
