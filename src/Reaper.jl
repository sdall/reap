
mutable struct ComparativeSetCoverRange{Sets,Set}
    sets::Sets
    tocover::Set
    alt::Set
end

comparesetcover(sets, set, alt) = ComparativeSetCoverRange{typeof(sets),typeof(set)}(sets, set, alt)

function iterate_impl(sc::ComparativeSetCoverRange{S,T}) where {S,T}
    w(u, v) = intersection_size(u, v) - 1
    i, g = nextcoverset(sc.sets, sc.tocover, w)
    if g <= 0
        return nothing
    else
        h = w(sc.tocover, sc.alt)
        setdiff!(sc.tocover, sc.sets[i])
        (g < h, sc)
    end
end
Base.IteratorSize(::Type{ComparativeSetCoverRange{A,B}}) where {A,B} = Base.SizeUnknown()
Base.eltype(::Type{ComparativeSetCoverRange{A,B}}) where {A,B} = Bool
Base.iterate(sc::ComparativeSetCoverRange{A,B}) where {A,B} = iterate_impl(sc)
Base.iterate(sc::ComparativeSetCoverRange{A,B}, _) where {A,B} = iterate_impl(sc)

count_usage(X, mask, y, pr, A) =
    sum(mask; init=0) do i
        x = copy!(A, X[i])
        any(comparesetcover(pr.factor, x, y)) || (!isempty(x) && intersection_size(y, x) >= 2)
    end

function are_factors_used(X, p, A::Vector{T}) where {T}
    isempty(p.factor) && return Vector{T}()
    U = falses(length(p.factor), Threads.nthreads())
    Threads.@threads for x in X
        greedy_setcover!(p.factor, copy!(A[Threads.threadid()], x), intersection_size) do i
            U[i, Threads.threadid()] = true
        end
        all(any(U; dims=2)) && break
    end
    reduce(|, U; dims=2)[:]
end

function create_mining_context(::Type{SetType}, max_factor_width) where {SetType}
    [SetType() for _ in 1:Threads.nthreads()],
    [SetType() for _ in 1:Threads.nthreads()],
    [MaxEntContext{SetType}() for _ in 1:Threads.nthreads()],
    [MaxEntContext{SetType}() for _ in 1:(max_factor_width + 1)]
end

"""
    fit(x[, y]; kwargs...) 
    
Discovers a concise set of informative patterns by using a the relaxed maximum entropy distribution and a pattern-usage-based Bayesian information criteria.

# Arguments

- `x::Vector{SetType}`: Input dataset.
- `y::Vector{Integer}`: Class labels that indicate subgroups in `x`.

# Options

- `min_support::integer`: Require a minimum support of each pattern.
- `max_factor_size::Integer`: Constraint the maximum number of patterns that each factor of the relaxed maximum entropy distribution can model. 
                                Note: As the per-factor inference complexity grows exponentially with the factor size, 
                                we ensure efficiency by limiting the size to be of at most `max_factor_size` (<= [`MAX_MAXENT_FACTOR_SIZE`](@ref) = 12).
- `max_factor_width::Integer`: Constraint the number of singletons per factor.
- `max_expansions::Integer`: Limit the number of search-space node-expansions per iteration. 
- `max_discoveries::Intege`: Terminate the algorithm after `max_discoveries` discoveries.
- `max_seconds::Float64`: Terminate the algorithm after approximately `max_seconds` seconds.
- `SetType::Type`: Underlying set type.

# Returns

Returns a relaxed maximum entropy distribution [`RelEnt`](@ref) which contains patterns, singletons, and estimated coefficients.
If `y` is provided, the function returns per-group distributions.

Note: Extract patterns (discoveries) via [`patterns`](@ref) or the per-group patterns via `patterns.`.

# Example

```julia-repl
julia> using Reap: fit, patterns
julia> p = fit(X)
julia> patterns(p)

julia> ps = fit(X, y)
julia> patterns.(ps)
```

"""
function fit(X::Vector{SetType}; min_support=2, max_factor_size=8, max_factor_width=50, args...) where {SetType<:Union{BitSet,Set}}
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    A, B, C, D = create_mining_context(SetType, max_factor_width)
    Dist = RelEnt{SetType,Float64}

    n = size(X, 1)
    L = Lattice{Candidate{SetType}}(X, x -> x.support)
    p = Dist([s.support / n for s in L.singletons])

    cost = log(n) / 2

    isforbidden(x) = Reap.isforbidden(p, x.set, max_factor_size, max_factor_width)
    discover_patterns!(L, x -> if x.support <= min_support || isforbidden(x)
                           0.0
                       else
                           tid = Threads.threadid()
                           u = count_usage(X, x.rows, x.set, p, A[tid])
                           u * (log(x.support / n) - log_expectation_ts!(p, x.set, A, B, C; tid=tid)::Float64) - cost::Float64
                       end,
                       isforbidden,
                       x -> insert_pattern!(p, x.support / n, x.set, max_factor_size, max_factor_width, D); args...)

    p.factor = p.factor[are_factors_used(X, p, A)]
    p
end

function fit(X::Vector{SetType}, y; min_support=2, max_factor_size=8, max_factor_width=50, args...) where {SetType<:Union{BitSet,Set}}
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    if y === nothing
        return fit(X; min_support=min_support, max_factor_size=max_factor_size, max_factor_width=max_factor_width, args...)
    end

    A, B, C, D = create_mining_context(SetType, max_factor_width)

    masks = [SetType(findall(==(i), y)) for i in unique(y)]
    n     = length.(masks)
    L     = Lattice{Candidate{SetType}}(X, x -> x.support)
    cost  = log(size(X, 1)) * length(masks) / 2

    fr(s, j) = intersection_size(s.rows, masks[j]) / n[j]
    Pr       = RelEnt{BitSet,Float64}
    p        = Pr[Pr(fr.(L.singletons, j)) for j in eachindex(masks)]

    isforbidden(x) = Reap.isforbidden(p, x.set, max_factor_size, max_factor_width)

    score(x) =
        if x.support < min_support || isforbidden(x)
            0.0
        else
            tid = Threads.threadid()
            sum(eachindex(p)) do i
                E = log_expectation_ts!(p[i], x.set, A, B, C; tid=tid)
                M = intersect(x.rows, masks[i])
                q = length(M) / n[i]
                q > 0 ? count_usage(X, M, x.set, p[i], A[tid]) * (log(q) - E) : 0.0
            end - cost
        end

    report(x) =
        mapreduce(|, eachindex(p)) do i
            tid = Threads.threadid()
            E = log_expectation_ts!(p[i], x.set, A, B, C; tid=tid)
            M = intersect(x.rows, masks[i])
            q = length(M) / n[i]
            h = q > 0 && (count_usage(X, M, x.set, p[i], A[tid]) * (log(q) - E) - log(n[i]) / 2 > 0)
            h && insert_pattern!(p[i], q, x.set, max_factor_size, max_factor_width, D)
        end

    discover_patterns!(L, score, isforbidden, report; args...)

    for i in eachindex(p)
        p[i].factor = p[i].factor[are_factors_used(X, p[i], A)]
    end

    p
end

function reap_estimate!(X::Vector{SetType}, S::Vector{Candidate{SetType}}, I::Vector{Candidate{SetType}};
                        min_support=2, max_factor_size=8, max_factor_width=50) where {SetType<:Union{BitSet,Set}}
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    A, B, C, D = create_mining_context(SetType, max_factor_width)
    Dist = RelEnt{SetType,Float64}

    n = size(X, 1)
    p = Dist(map(s -> s.support / n, I))

    cost = log(n) / 2
    skip = falses(length(S))

    score(x) =
        if x.support < min_support
            0.0
        else
            tid = Threads.threadid()
            u = count_usage(X, x.rows, x.set, p, A[tid])
            E = log_expectation_ts!(p, x.set, A, B, C; tid=tid)
            u * (log(x.support / n) - E) - cost
        end

    update!(y) =
        ThreadsX.foreach(enumerate(S)) do (i, x)
            if skip[i]
                x.score = 0
            elseif intersects(x.set, y.set)
                x.score = score(x)
            end
        end

    update!() =
        ThreadsX.foreach(S) do x
            x.score = score(x)
        end

    update!()

    while true
        h, i = findmax(u -> u.score, S)

        if h <= 0
            break
        end

        insert_pattern!(p, S[i].support / n, S[i].set, max_factor_size, max_factor_width, D)
        update!(S[i])
        skip[i] = true
    end

    p
end

function reap_estimate!(X::Vector{SetType}, y::Vector{Int}, S::Vector{Candidate{SetType}}, I::Vector{Candidate{SetType}};
                        min_support=2, max_factor_size=8, max_factor_width=50) where {SetType<:Union{BitSet,Set}}
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    A, B, C, D = create_mining_context(SetType, max_factor_width)

    masks = [SetType(findall(==(i), y)) for i in unique(y)]
    n     = length.(masks)

    fr(s, j) = intersection_size(s.rows, masks[j]) / n[j]
    Pr       = RelEnt{SetType,Float64}
    p        = Pr[Pr(fr.(I, j)) for j in eachindex(masks)]

    cost = log(size(X, 1)) * length(masks) / 2
    skip = falses(length(S))

    score(x) =
        if x.support < min_support
            0.0
        else
            tid = Threads.threadid()
            sum(eachindex(masks)) do i
                M = intersect(x.rows, masks[i])
                u = count_usage(X, M, x.set, p[i], A[tid])
                E = log_expectation_ts!(p[i], x.set, A, B, C; tid=tid)
                q = length(M) / n[i]
                q > 0 ? u * (log(length(M) / n[i]) - E) : 0
            end - cost
        end

    update!(y) =
        ThreadsX.foreach(enumerate(S)) do (i, x)
            if skip[i]
                x.score = 0
            elseif intersects(x.set, y.set)
                x.score = score(x)
            end
        end

    update!() =
        ThreadsX.foreach(S) do x
            x.score = score(x)
        end

    update!()

    while true
        h, i = findmax(u -> u.score, S)

        if h <= 0
            break
        end

        skip[i] = true
        x = S[i]

        r = ThreadsX.map(eachindex(p)) do j
            tid = Threads.threadid()
            M = intersect(x.rows, masks[j])
            q = length(M) / n[j]
            if q > 0
                E = log_expectation_ts!(p[j], x.set, A, B, C; tid=tid)
                u = count_usage(X, M, x.set, p[j], A[tid])
                q, q > 0 ? u * (log(q) - E) - log(n[j]) / 2 : 0.0
            else
                q, 0.0
            end
        end

        for j in eachindex(p)
            q, h = r[j]
            h > 0 && insert_pattern!(p[j], q, x.set, max_factor_size, max_factor_width, D)
        end

        update!(x)
    end
    p
end

"""
    reap_estimate

Greedily fits a relaxed maximum entropy distribution to dataset X[, y] from a given patternset.

# Arguments

- `x::Vector{SetType}`: Input dataset.
- `patterns::Vector{SetType}`: Set of to-be-fitted patterns.
- `y::Vector{Integer}`: Class labels that indicate subgroups in `x`.

# Options

- `min_support::Integer`: Require a minimum support of each pattern.
- `max_factor_size::Integer`: Constraint the maximum number of patterns that each factor of the maximum entropy distribution can model. 
                              Note: As the per-factor inference complexity grows exponentially with the factor size, 
                              we ensure efficiency by limiting the size to be of at most `max_factor_size` (<= [`MAX_MAXENT_FACTOR_SIZE`](@ref) = 12).
- `max_factor_width::Integer`: Constraint the number of singletons per factor.

# Returns

Returns a relaxed maximum entropy distribution [`RelEnt`](@ref) which contains patterns, singletons, and estimated coefficients.
If `y` is provided, returns per-group distributions.

# Example

```julia-repl
julia> using Reap: reap_estimate
julia> p = reap_estimate(X, patternset, y)
```

"""
function reap_estimate(X::Vector{S}, patterns::Vector{S}, y=nothing; kw...) where {S<:Union{BitSet,Set}}
    I = init_singletons(S, X)
    function candidate(x, I)
        rows = reduce(intersect, I[e].rows for e in x)
        Candidate{S}(x, rows, length(rows), 0)
    end
    C = [candidate(x, I) for x in patterns]
    if isnothing(y)
        reap_estimate!(X, C, I; kw...)
    else
        reap_estimate!(X, y, C, I; kw...)
    end
end
