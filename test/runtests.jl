using Test
using LVMaxTime_lite

@testset "LVMaxTime_lite" begin

    # ── 1. Types & Utilities ─────────────────────────────────────────────────
    @testset "Types & Utils" begin
        p = LVParams(1.0, 0.5, 0.5, 1.0)
        @test p.α == 1.0 && p.β == 0.5 && p.δ == 0.5 && p.γ == 1.0

        # Equilibrium: x₁* = γ/δ = 2.0, x₂* = α/β = 2.0
        x₁s, x₂s = equilibrium(p)
        @test x₁s == 2.0 && x₂s == 2.0

        # Pack / unpack
        v = params_to_vector(p)
        @test v == [1.0, 0.5, 0.5, 1.0]
        @test params_from_vector(v).α == p.α

        # RHS at equilibrium is zero (definition of equilibrium)
        dx₁, dx₂ = rhs(p, x₁s, x₂s)
        @test abs(dx₁) < 1e-14 && abs(dx₂) < 1e-14

        # det(J) at equilibrium is αγ (NOT zero!)
        # det(J) = αδ(γ/δ) + βγ(α/β) − αγ = αγ + αγ − αγ = αγ
        @test jacobian_det(p, x₁s, x₂s) ≈ p.α * p.γ

        # is_at_equilibrium
        @test is_at_equilibrium(p, x₁s, x₂s)
        @test !is_at_equilibrium(p, 1.5, 1.5)

        # Estimated period: 2π/√(αγ) = 2π for these params
        @test estimated_period(p) ≈ 2π
    end

    # ── 2. ODE Event Detection ───────────────────────────────────────────────
    @testset "ODE Event Detection" begin
        p = LVParams(1.0, 0.5, 0.5, 1.0)

        # Normal case: IC (1.5, 1.5) is NOT on any surface
        result = first_event_time(p, 1.5, 1.5)
        @test result.time > 0
        @test result.time < Inf
        @test result.event_type != EVT_NONE

        # Equilibrium → τ = Inf, no event
        x₁s, x₂s = equilibrium(p)
        result_eq = first_event_time(p, x₁s, x₂s)
        @test isinf(result_eq.time)
        @test result_eq.event_type == EVT_NONE

        # IC on E₁ (x₂ = α/β = 2.0, x₁ ≠ x₁*) → τ = 0
        result_e1 = first_event_time(p, 1.0, 2.0)
        @test result_e1.time == 0.0
        @test result_e1.event_type == EVT_DX1_ZERO

        # IC on E₂ (x₁ = γ/δ = 2.0, x₂ ≠ x₂*) → τ = 0
        result_e2 = first_event_time(p, 2.0, 1.0)
        @test result_e2.time == 0.0
        @test result_e2.event_type == EVT_DX2_ZERO

        # IC on E₃ (det(J)=0 line): (1,1) has αδ·1 + βγ·1 − αγ = 0.5+0.5−1 = 0
        result_e3 = first_event_time(p, 1.0, 1.0)
        @test result_e3.time == 0.0
        @test result_e3.event_type == EVT_DETJ_ZERO
    end

    # ── 3. Jacobian Determinant Properties ───────────────────────────────────
    @testset "Jacobian Properties" begin
        p = LVParams(1.0, 0.5, 0.5, 1.0)
        x₁s, x₂s = equilibrium(p)

        # det(J) at equilibrium = αγ
        @test jacobian_det(p, x₁s, x₂s) ≈ p.α * p.γ

        # det(J) = 0 defines the line x₁/x₁* + x₂/x₂* = 1
        # Check: 0.3·x₁* + 0.7·x₂* should give det(J)=0
        @test abs(jacobian_det(p, x₁s * 0.3, x₂s * 0.7)) < 1e-14

        # Below the line (origin side) → det(J) < 0
        @test jacobian_det(p, 0.5, 0.5) < 0
        # Above the line (equilibrium side) → det(J) > 0
        @test jacobian_det(p, 3.0, 3.0) > 0
    end

    # ── 4. Rigorous Certification ────────────────────────────────────────────
    @testset "Certification" begin
        # Params with a well-defined, moderate first event time
        p = LVParams(3.0, 1.0, 1.1, 2.5)
        x₁₀, x₂₀ = 2.0, 2.0

        cert = certify_event_time(p, x₁₀, x₂₀)
        @test cert.status == CERT_VERIFIED
        @test cert.tau_lower > 0
        @test cert.tau_upper > cert.tau_lower
        @test cert.certified_digits > 3.0
        @test cert.event_type != EVT_NONE

        # Float result should lie inside the certified bracket
        result = first_event_time(p, x₁₀, x₂₀)
        @test cert.tau_lower <= result.time <= cert.tau_upper

        # Surface safety dict has all 3 keys
        @test haskey(cert.surface_safety, "E1")
        @test haskey(cert.surface_safety, "E2")
        @test haskey(cert.surface_safety, "E3")

        # Edge case: equilibrium → CERT_FAILED
        p2 = LVParams(1.0, 0.5, 0.5, 1.0)
        cert_eq = certify_event_time(p2, equilibrium(p2)...)
        @test cert_eq.status == CERT_FAILED

        # Edge case: IC on surface (τ=0) → CERT_FAILED
        cert_surf = certify_event_time(p2, 1.0, 2.0)
        @test cert_surf.status == CERT_FAILED
    end

    # ── 5. certify_optimum (two-phase BoxCertificate) ─────────────────────────
    @testset "certify_optimum (BoxCertificate)" begin
        p = LVParams(3.0, 1.0, 1.1, 2.5)
        spec = LVProblemSpec(2.0, 2.0, (2.5, 3.0), (1.0, 1.1), (1.0, 1.1), (2.5, 3.0))
        result = first_event_time(p, 2.0, 2.0)
        opt = OptimizationResult(result, [result], 0.0, "test", nothing)

        cert = certify_optimum(opt, spec)
        @test cert isa BoxCertificate
        @test cert.status == CERT_VERIFIED
        @test cert.tau_lower > 0
        @test cert.tau_upper > cert.tau_lower
        @test cert.tau_lower <= result.time <= cert.tau_upper

        # Sub-certificates should be populated
        @test cert.safety_details isa Certificate
        @test cert.crossing_details isa Certificate
        @test cert.crossing_details.status == CERT_VERIFIED
    end

    # ── 6. certify_safety_box (box safety phase only) ──────────────────────────
    @testset "certify_safety_box" begin
        spec = LVProblemSpec(2.0, 2.0, (2.5, 3.0), (1.0, 1.1), (1.0, 1.1), (2.5, 3.0))

        # Get a τ_hint from a point evaluation at box midpoint
        p_mid = LVParams(2.75, 1.05, 1.05, 2.75)
        result = first_event_time(p_mid, 2.0, 2.0)
        τ_hint = result.time

        safety_cert = certify_safety_box(spec, τ_hint)
        @test safety_cert.tau_lower > 0
        @test safety_cert.status == CERT_SAFETY_ONLY  # box phase only proves safety

        # Safety bound should be less than or equal to the float hint (within time_factor)
        @test safety_cert.tau_lower <= τ_hint * 1.1
    end

end
