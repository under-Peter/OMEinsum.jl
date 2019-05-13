<!-- # OMEinsum -->
<div align="center"> <img
src="ome-logo.png"
alt="OMEinsum logo" width="510"></img>
<h1>OMEinsum - One More Einsum</h1>
</div>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://under-Peter.github.io/OMEinsum.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://under-Peter.github.io/OMEinsum.jl/dev)
[![Build Status](https://travis-ci.com/under-Peter/OMEinsum.jl.svg?branch=master)](https://travis-ci.com/under-Peter/OMEinsum.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/under-Peter/OMEinsum.jl?svg=true)](https://ci.appveyor.com/project/under-Peter/OMEinsum-jl)
[![Codecov](https://codecov.io/gh/under-Peter/OMEinsum.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/under-Peter/OMEinsum.jl)

This is a repository for the _Google Summer of Code_ project on Differentiable Tensor Networks.
It is a work in progress and will **change substantially this summer (2019)** - no guarantees can be made.

The goal is to implement an `einsum`-like function, see e.g. the Numpy documentation [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).
The ability to both define contractions at runtime and have efficient contractions if indices are shared between more than two tensors will set `OMEinsum` apart from the existing tools like
[`Einsum.jl`](https://github.com/ahwillia/Einsum.jl),
where contractions have to be written explicitly in the code,
and [`TensorOperations.jl`](https://github.com/Jutho/TensorOperations.jl),
which can not handle more general contractions.

Suggestions and Comments in the _Issues_ are welcome.

## License
MIT License
