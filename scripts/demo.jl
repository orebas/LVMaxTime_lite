#!/usr/bin/env julia
# ══════════════════════════════════════════════════════════════════════════════
# LVMaxTime — Minimum Critical Time for Lotka-Volterra Systems
# ══════════════════════════════════════════════════════════════════════════════
#
# Lotka-Volterra ODE:  dx₁/dt = αx₁ − βx₁x₂,  dx₂/dt = δx₁x₂ − γx₂
#
# Three "event surfaces" where qualitative dynamics change:
#   E₁: dx₁/dt = 0    (prey growth stalls)        ⟺  x₂ = α/β
#   E₂: dx₂/dt = 0    (predator growth stalls)     ⟺  x₁ = γ/δ
#   E₃: det(J) = 0    (Jacobian becomes singular)  ⟺  αδx₁ + βγx₂ = αγ
#
# Question: For fixed initial conditions, which parameters (α,β,δ,γ)
# within given bounds make the trajectory hit an event surface SOONEST?
#
# Pipeline:  ODE solve → global optimization → local refinement
#          → rigorous certification via Taylor model flowpipes
# ══════════════════════════════════════════════════════════════════════════════

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LVMaxTime_lite

# ── Example definitions ──────────────────────────────────────────────────────
# Each example has a different "flavor": tight bounds, wide bounds, asymmetric IC, etc.

struct ExampleDef
    name::String
    description::String
    spec::LVProblemSpec{Float64}
end

const EXAMPLES = [
    ExampleDef(
        "Near-nullcline",
        "IC close to E₁ — small τ expected",
        LVProblemSpec(1.5, 1.5,
            (0.8, 0.9), (0.5, 0.52), (0.8, 0.9), (0.5, 0.6))
    ),
    ExampleDef(
        "Symmetric wide",
        "IC moderately below equilibrium — moderate τ expected",
        LVProblemSpec(2.0, 2.0,
            (2.5, 3.0), (1.0, 1.1), (1.0, 1.1), (2.5, 3.0))
    ),
    ExampleDef(
        "Far from surfaces",
        "IC far above all event surfaces — larger τ expected",
        LVProblemSpec(5.0, 5.0,
            (3.0, 4.0), (1.0, 1.2), (1.0, 1.2), (3.0, 4.0))
    ),
    ExampleDef(
        "Asymmetric",
        "Asymmetric IC (0.5, 3.0) — interesting orbit geometry",
        LVProblemSpec(0.5, 3.0,
            (0.5, 0.8), (0.3, 0.35), (1.0, 1.2), (0.3, 0.4))
    ),
]

# ── Solver configuration ─────────────────────────────────────────────────────
# Faster settings for the demo (fewer evaluations than a production run)

const OPT_CONFIG = SolverConfig(
    abstol = 1e-12,
    reltol = 1e-12,
    maxiters = 1_000_000,
    max_time_factor = 5.0,
    optimizer_population = 50,
    optimizer_maxevals = 5_000,
    local_refinement_candidates = 5,
)

# ── Main pipeline ─────────────────────────────────────────────────────────────

function main()
    println("=" ^ 70)
    println("Rigorous Flowpipe Certification of Minimum Event Times")
    println("=" ^ 70)

    results = []

    for (i, ex) in enumerate(EXAMPLES)
        println("\n" * "=" ^ 70)
        println("Example $i: $(ex.name)")
        println("  $(ex.description)")
        println("=" ^ 70)

        spec = ex.spec
        println("  IC: ($(spec.x₁₀), $(spec.x₂₀))")
        println("  α ∈ $(spec.α_bounds), β ∈ $(spec.β_bounds)")
        println("  δ ∈ $(spec.δ_bounds), γ ∈ $(spec.γ_bounds)")

        # Step 1: Find the parameter set that minimizes the first event time
        println("\n  [1/2] Running optimization...")
        t_opt_start = time()
        opt = optimize_min_event_time(spec; config=OPT_CONFIG, verbose=false)
        t_opt = time() - t_opt_start

        p = opt.best.params
        println("    τ* = $(opt.best.time)")
        println("    Event: $(opt.best.event_type)")
        println("    Params: (α=$(round(p.α, digits=6)), β=$(round(p.β, digits=6)), " *
                "δ=$(round(p.δ, digits=6)), γ=$(round(p.γ, digits=6)))")
        println("    Optimization time: $(round(t_opt, digits=2))s")

        # Step 2: Rigorously certify (two-phase: box safety + point crossing)
        println("\n  [2/3] Running box safety certification (9D augmented)...")
        println("  [3/3] Running point crossing certification (2D at θ*)...")
        cert = certify_optimum(opt, spec)

        println("    Status: $(cert.status)")
        if cert.status == CERT_VERIFIED || cert.status == CERT_SAFETY_ONLY
            println("    τ_lower (box safety):    $(cert.tau_lower)")
            println("    τ_upper (point crossing): $(cert.tau_upper)")
            println("    τ* ∈ [$(cert.tau_lower), $(cert.tau_upper)]")
            println("    Bracket width: $(cert.tau_upper - cert.tau_lower)")
            println("    Certified digits: $(round(cert.certified_digits, digits=1))")
            println("    Float in bracket: $(cert.tau_lower <= opt.best.time <= cert.tau_upper)")
        end
        println("    Box phase time:   $(round(cert.safety_details.wall_time, digits=3))s")
        println("    Point phase time: $(round(cert.crossing_details.wall_time, digits=3))s")
        println("    Total cert time:  $(round(cert.wall_time, digits=3))s")

        # Surface safety report (from box phase)
        println("    Box surface safety (all θ ∈ Θ):")
        for name in ["E1", "E2", "E3"]
            status_str = cert.surface_safety[name] ? "safe" : "uncertain"
            println("      $name: $status_str")
        end

        push!(results, (opt=opt, cert=cert, t_opt=t_opt))
    end

    # ── Summary table — the money shot ────────────────────────────────────────
    println("\n" * "=" ^ 70)
    println("CERTIFICATION SUMMARY")
    println("=" ^ 70)
    println()
    println("  Ex  Status         Event          Float τ*       " *
            "Certified interval                          Width       Digits  Time")
    println("  " * "-" ^ 110)

    for (i, ex) in enumerate(EXAMPLES)
        r = results[i]
        cert = r.cert
        τ_float = round(r.opt.best.time, sigdigits=6)
        status_str = rpad(string(cert.status), 15)
        evt_str = rpad(string(cert.event_type), 15)

        if cert.status == CERT_VERIFIED
            bracket = cert.tau_upper - cert.tau_lower
            iv_str = rpad("[$(round(cert.tau_lower, sigdigits=8)), " *
                          "$(round(cert.tau_upper, sigdigits=8))]", 42)
            w_str = rpad(string(round(bracket, sigdigits=2)), 12)
            d_str = rpad("$(round(cert.certified_digits, digits=1))", 8)
            t_str = "$(round(cert.wall_time, digits=2))s"
            println("  $i   $status_str$evt_str$(rpad(string(τ_float), 15))$iv_str$w_str$d_str$t_str")
        else
            println("  $i   $status_str$evt_str$(rpad(string(τ_float), 15))—")
        end
    end

    # ── Final verification checks ─────────────────────────────────────────────
    println("\n  Verification:")
    all_verified = all(r.cert.status == CERT_VERIFIED for r in results)
    all_in_bracket = all(
        r.cert.tau_lower <= r.opt.best.time <= r.cert.tau_upper
        for r in results if r.cert.status == CERT_VERIFIED
    )
    all_good_digits = all(
        r.cert.certified_digits >= 3.5
        for r in results if r.cert.status == CERT_VERIFIED
    )

    println("    All examples CERT_VERIFIED: $(all_verified ? "YES" : "NO")")
    println("    All float τ* in bracket:    $(all_in_bracket ? "YES" : "NO")")
    println("    All >= 3.5 certified digits: $(all_good_digits ? "YES" : "NO")")
    println()
end

main()
