module Reap

include.(("SetUtils.jl", "MaxEnt.jl", "Lattice.jl", "Reaper.jl"))

_convert_dataset(xs::Vector{SetType}) where {SetType<:Union{Set,BitSet}} = xs
_convert_dataset(xs) = ThreadsX.map(x -> findall(!=(0), x) |> BitSet, eachrow(xs))

reap(X, args...; kwargs...) = fit(_convert_dataset(X), args...; kwargs...)

export reap, reap_estimate, patterns

end # module Reaper
