const MAX_MAXENT_FACTOR_SIZE = 12

Base.@kwdef mutable struct MaxEntFactor{SetType,FloatType}
    range::SetType                         = SetType()
    theta0::FloatType                      = 0
    singleton::Vector{Int64}               = []
    singleton_theta::Vector{FloatType}     = []
    singleton_frequency::Vector{FloatType} = []
    set::Vector{SetType}                   = []
    theta::Vector{FloatType}               = []
    frequency::Vector{FloatType}           = []
end

Base.insert!(f::MaxEntFactor{S,T}, q, x::S) where {S,T} =
    if !any(y -> y == x, f.set)
        union!(f.range, copy(x))
        push!(f.set, copy(x))
        push!(f.theta, copy(q))
        push!(f.frequency, copy(q))
    end

Base.insert!(f::MaxEntFactor{S,T}, q, x::Integer) where {S,T} =
    if !any(y -> y == x, f.singleton)
        push!(f.range, copy(x))
        push!(f.singleton, copy(x))
        push!(f.singleton_theta, copy(q))
        push!(f.singleton_frequency, copy(q))
    end

function log_probability(f::MaxEntFactor, x)
    prob = log(f.theta0)
    @fastmath @inbounds for i in eachindex(f.set)
        if issubset(f.set[i], x)
            prob += log(f.theta[i])
        end
    end
    @fastmath @inbounds for i in eachindex(f.singleton)
        if issubset(f.singleton[i], x)
            prob += log(f.singleton_theta[i])
        end
    end
    prob
end

function probability(f::MaxEntFactor, x)
    prob = f.theta0
    @fastmath @inbounds for i in eachindex(f.set)
        if issubset(f.set[i], x)
            prob *= f.theta[i]
        end
    end
    @fastmath @inbounds for i in eachindex(f.singleton)
        if issubset(f.singleton[i], x)
            prob *= f.singleton_theta[i]
        end
    end
    prob
end

function next_permutation(v::Int64)
    t = (v | (v - 1)) + 1
    t | ((div((t & -t), (v & -v)) >> 1) - 1)
end

function permute_level(n::Int64, k::Int64, fn)
    i = Int64(0)
    @fastmath @simd for z in 0:(k - 1)
        i |= (1 << z)
    end
    max = i << (n - k)
    while i != max
        fn(i)
        i = next_permutation(i)
    end
    fn(i) # max
end

function permute_all(fn, n::Int64)
    i = Int64(0)
    @fastmath @simd for z in 0:(n - 1)
        i |= (1 << z)
    end
    fn(i)
    @fastmath @simd for s in (n - 1):-1:1
        permute_level(n, s, fn)
    end
    fn(0)
end

function foreachbit(fn, i)
    l = i
    j = 0
    @fastmath @inbounds while l != 0
        Bool(l & 1) && fn(j)
        l >>= 1
        j = j + 1
    end
end

function create_permutation_list()
    perms = [zeros(Int64, 2^l) for l in 1:MAX_MAXENT_FACTOR_SIZE]
    for (n, class) in enumerate(perms)
        j = 1
        permute_all(n) do i
            class[j] = i
            j = j + 1
        end
    end
    perms
end

const permutations::Vector{Vector{Int64}} = create_permutation_list()
mutable struct MaxEntContext{S}
    classes::Vector{S}
    values::Vector{Int}
    MaxEntContext{S}(N=2^MAX_MAXENT_FACTOR_SIZE) where {S} = new{S}([S() for _ in 1:N], zeros(Int, N))
end

function equivalence_classes!(ctx::MaxEntContext{S}, width, itemsets, add_itemset=nothing) where {S}
    classes, values = ctx.classes, ctx.values

    len = length(itemsets) + (add_itemset !== nothing)

    if len == 0 || width == 0
        empty!(classes[1])
        values[1] = exp2(width)
        return 0
    end

    getset = ifelse(add_itemset !== nothing,
                    i -> (i <= length(itemsets) ? itemsets[i] : add_itemset),
                    i -> itemsets[i])

    @fastmath @inbounds for (index, p) in enumerate(permutations[len])
        empty!(classes[index])
        foreachbit(p) do i
            union!(classes[index], getset(i + 1))
        end
        k = exp2(width - length(classes[index]))
        values[index] = k

        @inbounds for prev in 1:(index - 1)
            if values[prev] > 0 && issubset(classes[index], classes[prev])
                if length(classes[index]) == length(classes[prev])
                    values[index] = 0
                    break
                else
                    values[index] -= values[prev]
                end
            end
        end
    end
    length(permutations[len])
end

function expectation_known(f::MaxEntFactor{S,T}, x, classes, values, limit::Int=length(classes))::T where {S,T}
    @fastmath sum(1:limit) do i
        @inbounds values[i] != 0 && issubset(x, classes[i]) ? values[i] * probability(f, classes[i]) : zero(T)
    end
end

expectation_known(f::MaxEntFactor, x, ctx::MaxEntContext) = expectation_known(f, x, ctx.classes, ctx.values)

function expectation_unknown(f::MaxEntFactor, x, ctx::MaxEntContext)
    u = equivalence_classes!(ctx, union_size(f.range, x), f.set, x)
    expectation_known(f, x, ctx.classes, ctx.values, u)
end

expectation(f::MaxEntFactor, x, ctx::MaxEntContext) =
    if (i = findfirst(isequal(x), f.set)) !== nothing
        f.frequency[i]
    else
        expectation_unknown(f, x, ctx)
    end

function iterative_scaling!(model::MaxEntFactor, ctx, ctx_lim, sctx, sctx_lim; max_iter=300, sensitivity=1e-4, epsilon=1e-5)
    bad_scaling_factor(a) = isinf(a) || isnan(a) || a <= 0 || (a != a)

    model.theta0 = exp2(-length(model.range) - 0)
    model.theta .= model.frequency .* model.theta0
    model.singleton_theta .= model.singleton_frequency .* model.theta0

    tol = sensitivity * (length(model.singleton) + length(model.set))
    pg  = typemax(model.theta0)
    for _ in 1:max_iter
        g = 0.0
        @fastmath @inbounds for i in eachindex(model.singleton)
            q = model.singleton_frequency[i]
            p = expectation_known(model, model.singleton[i], sctx[i].classes, sctx[i].values, sctx_lim[i])
            g += abs(q - p)
            if abs(q - p) < sensitivity || bad_scaling_factor(model.singleton_theta[i] * (q / p))
                continue
            end
            model.singleton_theta[i] *= q / p
        end
        @fastmath @inbounds for i in eachindex(model.set)
            q = model.frequency[i]
            p = expectation_known(model, model.set[i], ctx.classes, ctx.values, ctx_lim)
            g += abs(q - p)
            if abs(q - p) < sensitivity || bad_scaling_factor(model.theta[i] * (q / p))
                continue
            end
            model.theta[i] *= q / p
        end
        if g < tol || abs(g - pg) < epsilon
            pg = g
            break
        end
        pg = g
    end
    pg
end

function fit!(f::MaxEntFactor{S,T}, ctx::Vector{MaxEntContext{S}}) where {S,T}
    @assert length(ctx) >= length(f.singleton) + 1
    d = length(f.range)
    u = equivalence_classes!(ctx[end], d, f.set)
    U = ThreadsX.map(i -> equivalence_classes!(ctx[i], d, f.set, f.singleton[i]), eachindex(f.singleton))::Vector{Int64}
    iterative_scaling!(f, ctx[end], u, ctx, U)
end

abstract type MaxEntModel end

mutable struct RelEnt{S,T} <: MaxEntModel
    factor                  :: Vector{MaxEntFactor{S,T}}
    singleton               :: Vector{Int64}
    singleton_frequency     :: Vector{T}
    singleton_log_frequency :: Vector{T}

    RelEnt{S,T}(frequencies::Vector{T}) where {S,T} = new{S,T}([], collect(eachindex(frequencies)), frequencies, map(log, frequencies))
end

function create_factor(pr::RelEnt{S,T}, factors::Vector{Int}, singletons) where {S,T}
    next = MaxEntFactor{S,T}()
    for i in factors
        f = pr.factor[i]
        for j in eachindex(f.set)
            insert!(next, f.frequency[j], copy(f.set[j]))
        end
        for j in eachindex(f.singleton)
            insert!(next, f.singleton_frequency[j], f.singleton[j])
        end
    end
    for i in singletons
        insert!(next, pr.singleton_frequency[i], i)
    end
    next
end

intersection_size(u, v::MaxEntFactor) = intersection_size(u, v.range)
intersection_size(u::MaxEntFactor, v) = intersection_size(u.range, v)
Base.setdiff!(u::AbstractSet, v::MaxEntFactor) = Base.setdiff!(u, v.range)

@fastmath @inbounds function log_expectation!(pr::RelEnt{S,T}, x::S, buffer::S, ctx::MaxEntContext{S}) where {S,T}
    p = greedy_setcover!(+, zero(T), pr.factor, x, (u, v) -> intersection_size(u, v) - 1) do i
        copy!(buffer, x)
        intersect!(buffer, pr.factor[i].range)
        log(expectation(pr.factor[i], buffer, ctx))
    end
    p + sum(i -> pr.singleton_log_frequency[i], x; init=zero(T))
end
@fastmath @inbounds function expectation!(pr::RelEnt{S,T}, x::S, buffer::S, ctx::MaxEntContext{S}) where {S,T}
    p = greedy_setcover!(*, one(T), pr.factor, x, (u, v) -> intersection_size(u, v) - 1) do i
        copy!(buffer, x)
        intersect!(buffer, pr.factor[i].range)
        expectation(pr.factor[i], buffer, ctx)
    end
    p * prod(i -> pr.singleton_frequency[i], x; init=one(T))
end

function create_factor_greedily(pr::RelEnt{S,T}, factors::Vector{Int64}, singletons, max_size::Int64, max_width::Int64, q::T, x::S,
                                ctx::Vector{MaxEntContext{S}}) where {S,T}
    if length(x) > max_width
        return pr
    end
    next = MaxEntFactor{S,T}()
    for i in x
        insert!(next, pr.singleton_frequency[i], i)
    end
    insert!(next, q, deepcopy(x))
    fit!(next, ctx)
    used = falses(length(pr.factor), max_size)
    @fastmath @inbounds for _ in eachindex(pr.factor)
        if length(next.set) >= max_size
            break
        end
        best = (length(pr.factor), 0, 0.0)
        for i in factors, j in eachindex(pr.factor[i].set)
            if used[i, j]
                continue
            elseif !intersects(pr.factor[i].set[j], x) || union_size(pr.factor[i].set[j], next.range) >= max_width
                used[i, j] = true
            else
                p = expectation(next, pr.factor[i].set[j], ctx[end])
                q = pr.factor[i].frequency[j]
                g = q == 0 || p == 0 ? 0.0 : (q * log(q / p)) + (p * log(p / q))
                if best[3] < g
                    best = (i, j, g)
                end
            end
        end

        if best[3] > 0
            used[best[1], best[2]] = true
            f = pr.factor[best[1]]
            for s in f.set[best[2]]
                insert!(next, pr.singleton_frequency[s], s)
            end
            insert!(next, f.frequency[best[2]], copy(f.set[best[2]]))
            fit!(next, ctx)
        else
            break
        end
    end
    next
end

function insert_pattern!(p::RelEnt{S,T}, frequency::T, t::S, max_size, max_width, ctx::Vector{MaxEntContext{S}}) where {S,T}
    remaining_singletons = copy(t)
    factor_selection = Vector{Int}()
    greedy_setcover!(p.factor, remaining_singletons, intersection_size) do i
        push!(factor_selection, i)
    end
    next = create_factor_greedily(p, factor_selection, remaining_singletons, div(max_size, 2), max_width, frequency, copy(t), ctx)
    push!(p.factor, next)
    true
end

patterns(p::MaxEntModel) = unique([s for f in p.factor for s in f.set])

function log_expectation_ts!(pr::RelEnt, x, xbuffer, buffer, ctx; tid=Threads.threadid())
    log_expectation!(pr, copy!(xbuffer[tid], x), buffer[tid], ctx[tid])
end
function expectation_ts!(pr::RelEnt, x, xbuffer, buffer, ctx; tid=Threads.threadid())
    expectation!(pr, copy!(xbuffer[tid], x), buffer[tid], ctx[tid])
end

isallowed(::RelEnt, t, max_size, max_width) = length(t) <= max_width
isforbidden(p::RelEnt, x, max_size, max_width) = !isallowed(p, x, max_size, max_width)
isforbidden(P::Vector{RelEnt{S, T}}, x, max_size, max_width) where {S, T} = !all(P) do p
    isallowed(p, x, max_size, max_width)
end
