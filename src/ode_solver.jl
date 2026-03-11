# ode_solver.jl — ODE integration with event-surface detection
#
# Integrates the Lotka-Volterra system forward in time using a high-order
# Runge-Kutta solver (Vern9) with continuous callbacks that detect when the
# trajectory crosses any of the three event surfaces.

using DifferentialEquations

"""
    first_event_time(p, x₁₀, x₂₀; config=SolverConfig()) → EventResult

Integrate the LV ODE from (x₁₀, x₂₀) and return the time τ when the trajectory
first crosses an event surface:
  E₁: dx₁/dt = 0  ⟺  x₂ = α/β
  E₂: dx₂/dt = 0  ⟺  x₁ = γ/δ
  E₃: det(J) = 0  ⟺  αδx₁ + βγx₂ = αγ

Special cases:
- IC at equilibrium → τ = Inf (trajectory never leaves)
- IC already on a surface → τ = 0
"""
function first_event_time(p::LVParams, x₁₀::Real, x₂₀::Real;
                          config::SolverConfig=SolverConfig())
    # Handle degenerate cases before launching the solver
    if is_at_equilibrium(p, x₁₀, x₂₀)
        return EventResult(Inf, EVT_NONE, (Float64(x₁₀), Float64(x₂₀)), p)
    end
    on_surface, evt = is_on_event_surface(p, x₁₀, x₂₀)
    if on_surface
        return EventResult(0.0, evt, (Float64(x₁₀), Float64(x₂₀)), p)
    end

    # Integration time: several periods to be safe
    T_est = estimated_period(p)
    T_max = config.max_time_factor * T_est
    u0 = [Float64(x₁₀), Float64(x₂₀)]

    # The Lotka-Volterra right-hand side (in-place form for DifferentialEquations)
    function lv!(du, u, params, t)
        α, β, δ, γ = params
        du[1] = u[1] * (α - β * u[2])
        du[2] = u[2] * (δ * u[1] - γ)
        nothing
    end

    params_vec = (p.α, p.β, p.δ, p.γ)

    # ── Event condition functions ─────────────────────────────────────────
    # Each returns a scalar whose zero-crossing triggers the callback.

    # E₁: dx₁/dt = x₁(α − βx₂) = 0.  Since x₁>0, this means α − βx₂ = 0.
    condition_E1(u, t, integrator) = integrator.p[1] - integrator.p[2] * u[2]

    # E₂: dx₂/dt = x₂(δx₁ − γ) = 0.  Since x₂>0, this means δx₁ − γ = 0.
    condition_E2(u, t, integrator) = integrator.p[3] * u[1] - integrator.p[4]

    # E₃: det(J) = αδx₁ + βγx₂ − αγ = 0.
    function condition_E3(u, t, integrator)
        α, β, δ, γ = integrator.p
        return α * δ * u[1] + β * γ * u[2] - α * γ
    end

    # ── Callback construction ─────────────────────────────────────────────
    # A tiny guard time rejects spurious t≈0 triggers from the solver.
    # True t=0 events are already handled by is_on_event_surface above.
    t_min_guard = T_est * 1e-12

    # Mutable storage for the first detected event
    event_info = Ref{Tuple{Float64,EventType,Float64,Float64}}(
        (T_max, EVT_NONE, Float64(x₁₀), Float64(x₂₀))
    )

    function make_affect(evt_type)
        function affect!(integrator)
            t = integrator.t
            if t > t_min_guard
                event_info[] = (t, evt_type, integrator.u[1], integrator.u[2])
                terminate!(integrator)
            end
        end
        return affect!
    end

    # ContinuousCallback with BOTH upcrossing and downcrossing affects.
    # Why both?  The trajectory orbits, so a surface value can cross zero
    # in either direction depending on which quadrant the orbit enters.
    cb1 = ContinuousCallback(condition_E1, make_affect(EVT_DX1_ZERO),
                              make_affect(EVT_DX1_ZERO); abstol=1e-14)
    cb2 = ContinuousCallback(condition_E2, make_affect(EVT_DX2_ZERO),
                              make_affect(EVT_DX2_ZERO); abstol=1e-14)
    cb3 = ContinuousCallback(condition_E3, make_affect(EVT_DETJ_ZERO),
                              make_affect(EVT_DETJ_ZERO); abstol=1e-14)

    cb = CallbackSet(cb1, cb2, cb3)

    # ── Solve ─────────────────────────────────────────────────────────────
    prob = ODEProblem(lv!, u0, (0.0, T_max), params_vec)
    sol = solve(prob, Vern9();
                callback=cb,
                abstol=config.abstol,
                reltol=config.reltol,
                maxiters=config.maxiters,
                save_everystep=false,
                dense=true)

    t_event, evt_type, x₁_event, x₂_event = event_info[]
    return EventResult(t_event, evt_type, (x₁_event, x₂_event),
                       LVParams(p.α, p.β, p.δ, p.γ))
end
