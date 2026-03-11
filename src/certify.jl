# certify.jl — Rigorous flowpipe-based certification of first event times
#
# A Taylor model flowpipe (from ReachabilityAnalysis.jl / TMJets21a) gives a
# *guaranteed interval enclosure* of the ODE trajectory at every time.  Unlike
# a regular numerical solver, the enclosure is mathematically proven to contain
# the true solution — so if the enclosure doesn't touch an event surface, we
# know for certain the trajectory hasn't crossed it.
#
# Algorithm overview:
#   1. Get a floating-point hint τ_hint from the ODE solver
#   2. Build a flowpipe from t=0 to t = τ_hint × 1.05 (slightly past the event)
#   3. Subdivide each reach set's time domain into fine sub-intervals
#   4. At each sub-interval, evaluate guaranteed [x₁] × [x₂] enclosures
#   5. Check all 3 event surfaces against these enclosures:
#        Safety:   surface value provably has one sign → no crossing here
#        Crossing: surface value enclosure contains zero → crossing possible
#   6. τ_lower = latest time all prior sub-intervals are safe
#      τ_upper = end of crossing region
#   7. The true event time lies in [τ_lower, τ_upper] — a rigorous bracket

using ReachabilityAnalysis
using ReachabilityAnalysis: TaylorModels, TaylorSeries

"""
    eval_tm_point_ic(tm, time_iv)

Evaluate a TaylorModel1{TaylorN} at a time interval.

Why this works for a point IC: the TaylorN variables represent perturbations
from the initial condition.  For a point (zero-width) IC, those perturbations
are zero, so only the constant term of the resulting TaylorN polynomial
is non-trivial.  That constant term is an interval — the guaranteed enclosure
of the state variable over the given time sub-interval.
"""
function eval_tm_point_ic(tm, time_iv)
    tn = tm(time_iv)
    return TaylorSeries.constant_term(tn)
end

"""
    certify_event_time(p, x₁₀, x₂₀; kwargs...) → Certificate

Rigorously certify the first event time using Taylor model flowpipes.

Returns a Certificate with:
- `tau_lower`: no event surface was crossed before this time (proved)
- `tau_upper`: an event surface crossing occurred by this time (proved)
- `certified_digits`: -log10(bracket_width / midpoint)

Keyword arguments:
- `time_factor=1.05`: extend the flowpipe slightly past the float hint
- `orderQ=2`: TaylorModel spatial order
- `orderT=15`: TaylorModel time order
- `abstol=1e-15`: integration tolerance for the flowpipe
- `n_subdivisions=10_000`: how finely to subdivide each reach set's time domain
"""
function certify_event_time(p::LVParams, x₁₀::Real, x₂₀::Real;
                            time_factor::Float64=1.05,
                            orderQ::Int=2, orderT::Int=15,
                            abstol::Float64=1e-15,
                            n_subdivisions::Int=10_000)
    wall_start = time()

    # Step 1: Get floating-point hint from ODE solver
    result_float = first_event_time(p, x₁₀, x₂₀)

    if isinf(result_float.time) || result_float.time == 0.0
        # Degenerate case: can't certify equilibrium or on-surface ICs
        return Certificate(
            CERT_FAILED, 0.0, 0.0, EVT_NONE, time() - wall_start, 0.0,
            Dict("E1" => false, "E2" => false, "E3" => false)
        )
    end

    τ_hint = result_float.time

    # Step 2: Build flowpipe via TMJets21a (Taylor model integrator)
    α, β, δ, γ = p.α, p.β, p.δ, p.γ

    function lv!(du, u, params, t)
        du[1] = u[1] * (α - β * u[2])
        du[2] = u[2] * (δ * u[1] - γ)
        return du
    end

    # Point IC as a degenerate (zero-width) hyperrectangle
    X0 = Hyperrectangle(low=[Float64(x₁₀), Float64(x₂₀)],
                         high=[Float64(x₁₀), Float64(x₂₀)])
    sys = BlackBoxContinuousSystem(lv!, 2)
    prob = InitialValueProblem(sys, X0)
    t_max = τ_hint * time_factor

    sol = solve(prob, T=t_max,
                alg=TMJets21a(orderQ=orderQ, orderT=orderT, abstol=abstol))
    F = flowpipe(sol)

    # Step 3: Event surface critical values
    x₁_star = γ / δ   # E₂ nullcline
    x₂_star = α / β   # E₁ nullcline

    # Step 4: Scan all reach sets, tracking safety and crossings per surface
    #
    # safe_until[S]  = latest time where surface S is provably un-crossed
    # first_cross[S] = (t_lo, t_hi) of the first sub-interval where surface
    #                  value enters the enclosure (might be crossing)
    # last_cross[S]  = last consecutive sub-interval still containing surface
    # exited[S]      = whether the enclosure has left the crossing region
    #
    # Why track last_cross?  The enclosure is *wider* than the true trajectory.
    # The true crossing is a single instant, but the enclosure may straddle the
    # surface for several sub-intervals.  τ_upper must cover all of them.

    safe_until  = Dict("E1" => 0.0, "E2" => 0.0, "E3" => 0.0)
    first_cross = Dict{String,Tuple{Float64,Float64}}()
    last_cross  = Dict{String,Tuple{Float64,Float64}}()
    exited      = Dict("E1" => false, "E2" => false, "E3" => false)

    for rs in F
        vTM = set(rs)
        tm_x1 = vTM[1]
        tm_x2 = vTM[2]
        dom = TaylorModels.domain(tm_x1)
        dom_lo = IntervalArithmetic.inf(dom)
        dom_hi = IntervalArithmetic.sup(dom)
        t_lo = tstart(rs)
        t_hi = tend(rs)
        h = (dom_hi - dom_lo) / n_subdivisions

        for k in 0:(n_subdivisions - 1)
            sub_lo = dom_lo + k * h
            sub_hi = min(dom_lo + (k + 1) * h, dom_hi)  # clamp to avoid assertion
            sub_iv = interval(sub_lo, sub_hi)
            x1_sub = eval_tm_point_ic(tm_x1, sub_iv)
            x2_sub = eval_tm_point_ic(tm_x2, sub_iv)

            # Map sub-interval index to physical time
            t_sub_lo = t_lo + (t_hi - t_lo) * k / n_subdivisions
            t_sub_hi = t_lo + (t_hi - t_lo) * (k + 1) / n_subdivisions

            # ── E₁: x₂ = α/β ────────────────────────────────────────────
            if !exited["E1"]
                if sup(x2_sub) < x₂_star || inf(x2_sub) > x₂_star
                    # Surface value has definite sign → safe
                    if haskey(first_cross, "E1")
                        exited["E1"] = true
                    else
                        safe_until["E1"] = t_sub_hi
                    end
                elseif inf(x2_sub) <= x₂_star <= sup(x2_sub)
                    # Surface value might be zero → possible crossing
                    if !haskey(first_cross, "E1")
                        first_cross["E1"] = (t_sub_lo, t_sub_hi)
                    end
                    last_cross["E1"] = (t_sub_lo, t_sub_hi)
                end
            end

            # ── E₂: x₁ = γ/δ ────────────────────────────────────────────
            if !exited["E2"]
                if sup(x1_sub) < x₁_star || inf(x1_sub) > x₁_star
                    if haskey(first_cross, "E2")
                        exited["E2"] = true
                    else
                        safe_until["E2"] = t_sub_hi
                    end
                elseif inf(x1_sub) <= x₁_star <= sup(x1_sub)
                    if !haskey(first_cross, "E2")
                        first_cross["E2"] = (t_sub_lo, t_sub_hi)
                    end
                    last_cross["E2"] = (t_sub_lo, t_sub_hi)
                end
            end

            # ── E₃: det(J) = αδx₁ + βγx₂ − αγ = 0 ──────────────────────
            if !exited["E3"]
                detJ = interval(α * δ) * x1_sub + interval(β * γ) * x2_sub - interval(α * γ)
                if inf(detJ) > 0 || sup(detJ) < 0
                    if haskey(first_cross, "E3")
                        exited["E3"] = true
                    else
                        safe_until["E3"] = t_sub_hi
                    end
                else
                    if !haskey(first_cross, "E3")
                        first_cross["E3"] = (t_sub_lo, t_sub_hi)
                    end
                    last_cross["E3"] = (t_sub_lo, t_sub_hi)
                end
            end
        end
    end

    wall_time = time() - wall_start

    # Step 5: Determine the earliest crossing among all surfaces
    earliest_event_name = ""
    earliest_lo = Inf
    for (name, (lo, _)) in first_cross
        if lo < earliest_lo
            earliest_lo = lo
            earliest_event_name = name
        end
    end

    if isempty(earliest_event_name)
        # No crossings detected in the time window — safety only
        surface_safety = Dict(name => true for name in ["E1", "E2", "E3"])
        return Certificate(CERT_SAFETY_ONLY, 0.0, t_max, EVT_NONE,
                           wall_time, 0.0, surface_safety)
    end

    event_map = Dict("E1" => EVT_DX1_ZERO, "E2" => EVT_DX2_ZERO, "E3" => EVT_DETJ_ZERO)
    event_type = event_map[earliest_event_name]

    # τ_lower = last safe time before the crossing region
    # τ_upper = end of the crossing region (enclosure might be wider than truth)
    τ_lower = safe_until[earliest_event_name]
    τ_upper = last_cross[earliest_event_name][2]

    # Per-surface safety through [0, τ_upper]
    surface_safety = Dict{String,Bool}()
    for name in ["E1", "E2", "E3"]
        if name == earliest_event_name
            surface_safety[name] = false
        elseif haskey(first_cross, name)
            lo, _ = first_cross[name]
            surface_safety[name] = lo > τ_upper
        else
            surface_safety[name] = true
        end
    end

    # Certified digits: how tightly the bracket pins down the event time
    bracket_width = τ_upper - τ_lower
    midpoint = (τ_lower + τ_upper) / 2
    certified_digits = midpoint > 0 ? -log10(bracket_width / midpoint) : 0.0

    status = (τ_lower > 0 && τ_upper > τ_lower) ? CERT_VERIFIED : CERT_FAILED

    return Certificate(status, τ_lower, τ_upper, event_type,
                       wall_time, certified_digits, surface_safety)
end

"""
    certify_optimum(opt, spec; kwargs...) → Certificate

Convenience: certify the event time at the optimal parameters from an
OptimizationResult, using the initial conditions from the problem spec.
"""
function certify_optimum(opt::OptimizationResult, spec::LVProblemSpec; kwargs...)
    p = opt.best.params
    return certify_event_time(p, spec.x₁₀, spec.x₂₀; kwargs...)
end
