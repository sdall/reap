# The Relaxed Maximum Entropy Distribution and its Application to Pattern Discovery

This repository provides a Julia library that implements a relaxed maximum entropy distribution (RelEnt) and the PAC-based RelEnt accelerated pattern discovery algorithm (Reap). It estimates a relaxed maximum entropy distribution by discovering sets of higher-order feature interactions (i.e., patterns) in Boolean data. For multiple groups in the data, Reap highlights differences and commonalities between the groups by leveraging associations between patterns and subsets of groups.

The code is a from-scratch implementation of algorithms described in the [paper](https://doi.org/10.1109/ICDM50108.2020.00112).

```
Sebastian Dalleiger and Jilles Vreeken. 2020. 
The Relaxed Maximum Entropy Distribution and its Application to Pattern Discovery. 
(ICDM '20), pp. 978–983, https://doi.org/10.1109/ICDM50108.2020.00112
```

Please consider [citing](CITATION.bib) the paper.

[Contributions](CONTRIBUTING.md) are welcome.

## Installation

To install the library from the REPL:
```julia-repl
julia> using Pkg; Pkg.add(url="https://github.com/sdall/reap.git")
```

To install the library from the command line:
```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/sdall/reap.git")'
```

To set up the command line interface (CLI) located in `bin/reap.jl`:

1. Clone the repository:
```sh
git clone https://github.com/sdall/reap
```
2. Install the required dependencies including the library:
```sh
julia -e 'using Pkg; Pkg.add(path="./reap"); Pkg.add.(["Comonicon", "CSV", "GZip", "JSON"])'
```

## Usage

For example, to fit a relaxed maximum entropy pattern distribution from a given pattern set:
```julia-repl
julia> using Reap: reap_estimate, reap, patterns
julia> p = reap_estimate(X, patternset; max_factor_size=5)
```

For example, to discover a relaxed maximum entropy pattern distribution from a given dataset:
```julia-repl
julia> p = reap(X)
julia> patterns(p)
```

To see the full list of options:
```julia-repl
help?> reap_estimate
help?> reap
```

A typical usage of the command line interface is:
```sh
chmod +x bin/reap.jl
bin/reap.jl dataset.dat.gz dataset.labels.gz > output.json
```
The output contains `patterns` and `executiontime` in seconds (cf. `--measure-time` for details). 
For further information regarding usage, available options, or input format, please see the documentation:
```sh
bin/reap.jl --help
```
