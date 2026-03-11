# optimize.jl — Two-stage parameter optimization
#
# Goal: find parameters (α, β, δ, γ) within given bounds that make the
# trajectory hit an event surface as SOON as possible.
#
# Why minimize?  The minimum over the parameter box gives the length of the
# guaranteed-safe interval — no event before this time for ANY parameters
# in the box.
#
# Two stages:
#   1. Global search with BlackBoxOptim (adaptive differential evolution)
#      — the landscape is non-convex with many local minima
#   2. Local refinement with Nelder-Mead on the best candidates
#      — squeezes out the last few digits of accuracy

using BlackBoxOptim
using Optim

"""
    optimize_min_event_time(spec; config=SolverConfig(), verbose=true) → OptimizationResult

Find parameters within `spec`'s bounds that minimize the first event time τ.

Stage 1: BlackBoxOptim global search (adaptive DE, population-based).
Stage 2: Nelder-Mead local refinement on the best candidate + perturbations.
"""
function optimize_min_event_time(spec::LVProblemSpec;
                                  config::SolverConfig=SolverConfig(),
                                  verbose::Bool=true)
    t_start = time()
    lb, ub = bounds_to_ranges(spec)

    # Large penalty for failed or degenerate evaluations
    penalty = 1e10

    # Objective: τ(θ) — the first event time at parameters θ = [α, β, δ, γ]
    function objective(θ)
        if any(θ .<= 0)
            return penalty
        end
        p = params_from_vector(θ)
        try
            result = first_event_time(p, spec.x₁₀, spec.x₂₀; config=config)
            if result.event_type == EVT_NONE || isinf(result.time)
                return penalty
            end
            return result.time
        catch e
            verbose && @warn "Evaluation failed at θ=$θ: $e"
            return penalty
        end
    end

    # ── Stage 1: Global optimization ──────────────────────────────────────
    if verbose
        println("Stage 1: Global optimization with BlackBoxOptim...")
        println("  Bounds: α∈$(spec.α_bounds), β∈$(spec.β_bounds), " *
                "δ∈$(spec.δ_bounds), γ∈$(spec.γ_bounds)")
    end

    search_range = [(lb[i], ub[i]) for i in 1:4]
    bbo_result = bboptimize(objective;
                             SearchRange=search_range,
                             Method=:adaptive_de_rand_1_bin_radiuslimited,
                             PopulationSize=config.optimizer_population,
                             MaxFuncEvals=config.optimizer_maxevals,
                             TraceMode=verbose ? :compact : :silent,
                             TargetFitness=-Inf)

    best_θ = best_candidate(bbo_result)
    best_val = best_fitness(bbo_result)

    if verbose
        println("\n  Global best: τ = $best_val")
        println("  At: α=$(best_θ[1]), β=$(best_θ[2]), δ=$(best_θ[3]), γ=$(best_θ[4])")
    end

    # ── Stage 2: Local refinement ─────────────────────────────────────────
    # Build a set of starting points: the global best + random perturbations
    candidates_θ = [best_θ]
    for _ in 1:(config.local_refinement_candidates - 1)
        perturbed = best_θ .* (1.0 .+ 0.05 .* randn(4))
        perturbed = clamp.(perturbed, lb, ub)
        push!(candidates_θ, perturbed)
    end

    if verbose
        println("\nStage 2: Local refinement with Nelder-Mead...")
    end

    for (i, θ₀) in enumerate(candidates_θ)
        try
            nm_result = Optim.optimize(
                objective, lb, ub, θ₀,
                Fminbox(NelderMead()),
                Optim.Options(iterations=5000, f_reltol=1e-14, x_abstol=1e-14)
            )
            val = Optim.minimum(nm_result)
            if val < best_val
                best_val = val
                best_θ = Optim.minimizer(nm_result)
                verbose && println("  Candidate $i improved: τ = $val")
            end
        catch e
            verbose && @warn "Local refinement failed for candidate $i: $e"
        end
    end

    # ── Build result ──────────────────────────────────────────────────────
    best_params = params_from_vector(best_θ)
    best_result = first_event_time(best_params, spec.x₁₀, spec.x₂₀; config=config)

    wall_time = time() - t_start

    if verbose
        println("\nOptimization complete in $(round(wall_time, digits=2))s")
        println("  Best τ = $(best_result.time)")
        println("  Event: $(best_result.event_type)")
        println("  Params: α=$(best_params.α), β=$(best_params.β), " *
                "δ=$(best_params.δ), γ=$(best_params.γ)")
    end

    return OptimizationResult(best_result, [best_result], wall_time, "ode", nothing)
end
