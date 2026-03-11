# types.jl — Data structures and utility functions for the LV critical-time problem
#
# Lotka-Volterra ODE:
#   dx₁/dt = α x₁ − β x₁ x₂    (prey)
#   dx₂/dt = δ x₁ x₂ − γ x₂    (predator)
#
# We study three "event surfaces" where qualitative dynamics change:
#   E₁: dx₁/dt = 0   ⟺  x₂ = α/β     (prey growth stalls)
#   E₂: dx₂/dt = 0   ⟺  x₁ = γ/δ     (predator growth stalls)
#   E₃: det(J) = 0   ⟺  αδx₁ + βγx₂ = αγ  (Jacobian singular)

# ── Data Structures ──────────────────────────────────────────────────────────

# The 4 parameters of the Lotka-Volterra system.
struct LVParams{T<:Real}
    α::T  # prey birth rate
    β::T  # predation rate (prey killed per predator encounter)
    δ::T  # predator efficiency (predator growth per prey eaten)
    γ::T  # predator death rate
end

# Promotion constructor: LVParams(1, 0.5, 0.5, 1) promotes all to Float64.
LVParams(α, β, δ, γ) = LVParams(promote(α, β, δ, γ)...)

# Full problem specification: initial conditions + box constraints on parameters.
struct LVProblemSpec{T<:Real}
    x₁₀::T                        # initial prey population
    x₂₀::T                        # initial predator population
    α_bounds::Tuple{T,T}           # allowed range for α
    β_bounds::Tuple{T,T}           # allowed range for β
    δ_bounds::Tuple{T,T}           # allowed range for δ
    γ_bounds::Tuple{T,T}           # allowed range for γ
end

# Solver/optimizer configuration with sensible defaults.
Base.@kwdef struct SolverConfig
    abstol::Float64 = 1e-12
    reltol::Float64 = 1e-12
    maxiters::Int = 1_000_000
    max_time_factor::Float64 = 5.0     # integrate up to 5× the estimated period
    optimizer_population::Int = 100     # BlackBoxOptim population size
    optimizer_maxevals::Int = 50_000    # max function evaluations in global search
    local_refinement_candidates::Int = 10  # Nelder-Mead restarts
end

# Which event surface was hit first.
@enum EventType begin
    EVT_DX1_ZERO    # x₂ crossed α/β  (prey growth stalls)
    EVT_DX2_ZERO    # x₁ crossed γ/δ  (predator growth stalls)
    EVT_DETJ_ZERO   # det(J) crossed zero  (Jacobian singular)
    EVT_NONE        # no event found (e.g. IC at equilibrium)
end

# Result of computing the first event time for one parameter set.
struct EventResult{T<:Real}
    time::T                  # τ — time of first surface crossing
    event_type::EventType    # which surface was hit
    state::Tuple{T,T}        # (x₁, x₂) at the event
    params::LVParams{T}      # the parameters that produced this result
end

# Result of the two-stage optimization (global + local).
struct OptimizationResult{T}
    best::EventResult{T}                        # best (minimum τ) found
    candidates::Vector{EventResult{T}}           # top candidates evaluated
    wall_time::Float64                           # seconds spent optimizing
    method::String                               # description of method used
    rigorous_bound::Union{Nothing,Tuple{T,T}}    # (lower, upper) if available
end

# Status of a rigorous certification attempt.
@enum CertificationStatus begin
    CERT_VERIFIED     # both safety (no crossing before τ_lower) and liveness (crossing by τ_upper)
    CERT_SAFETY_ONLY  # proved no crossing up to some time, but didn't detect a crossing
    CERT_FAILED       # certification could not be completed (e.g. degenerate IC)
end

# Rigorous certificate: a guaranteed bracket [τ_lower, τ_upper] for the first event time.
struct Certificate{T<:Real}
    status::CertificationStatus
    tau_lower::T               # guaranteed: no event before this time
    tau_upper::T               # guaranteed: event has occurred by this time
    event_type::EventType      # which surface crossed first
    wall_time::Float64         # certification wall-clock time (seconds)
    certified_digits::Float64  # -log10(bracket_width / midpoint)
    surface_safety::Dict{String,Bool}  # per-surface safety through [0, τ_upper]
end

# ── Utility Functions ────────────────────────────────────────────────────────

# Convert a 4-element vector [α, β, δ, γ] → LVParams.
params_from_vector(v) = LVParams(v[1], v[2], v[3], v[4])

# Convert LVParams → 4-element vector.
params_to_vector(p::LVParams) = [p.α, p.β, p.δ, p.γ]

# Interior equilibrium: x₁* = γ/δ, x₂* = α/β.
equilibrium(p::LVParams) = (p.γ / p.δ, p.α / p.β)

# Approximate period of small oscillations near equilibrium: T ≈ 2π/√(αγ).
estimated_period(p::LVParams) = 2π / sqrt(p.α * p.γ)

# Right-hand side of the LV ODE evaluated at (x₁, x₂).
function rhs(p::LVParams, x₁, x₂)
    dx₁ = x₁ * (p.α - p.β * x₂)
    dx₂ = x₂ * (p.δ * x₁ - p.γ)
    return (dx₁, dx₂)
end

# Determinant of the Jacobian ∂f/∂x at (x₁, x₂).
# Key fact: det(J) = αδx₁ + βγx₂ − αγ is LINEAR in state.
# At equilibrium, det(J) = αγ (positive — NOT zero!).
jacobian_det(p::LVParams, x₁, x₂) = p.α * p.δ * x₁ + p.β * p.γ * x₂ - p.α * p.γ

# Check whether (x₁, x₂) is at the interior equilibrium (within tolerance).
function is_at_equilibrium(p::LVParams, x₁, x₂; tol=1e-10)
    x₁s, x₂s = equilibrium(p)
    return abs(x₁ - x₁s) < tol * max(1.0, abs(x₁s)) &&
           abs(x₂ - x₂s) < tol * max(1.0, abs(x₂s))
end

# Check whether (x₁, x₂) lies on any event surface. Returns (is_on, which_surface).
function is_on_event_surface(p::LVParams, x₁, x₂; tol=1e-10)
    dx₁, dx₂ = rhs(p, x₁, x₂)
    if abs(dx₁) < tol * max(1.0, abs(x₁))
        return (true, EVT_DX1_ZERO)
    elseif abs(dx₂) < tol * max(1.0, abs(x₂))
        return (true, EVT_DX2_ZERO)
    elseif abs(jacobian_det(p, x₁, x₂)) < tol * max(1.0, p.α * p.γ)
        return (true, EVT_DETJ_ZERO)
    end
    return (false, EVT_NONE)
end

# Extract (lower_bounds, upper_bounds) vectors from a problem spec for the optimizer.
function bounds_to_ranges(spec::LVProblemSpec)
    lb = [spec.α_bounds[1], spec.β_bounds[1], spec.δ_bounds[1], spec.γ_bounds[1]]
    ub = [spec.α_bounds[2], spec.β_bounds[2], spec.δ_bounds[2], spec.γ_bounds[2]]
    return (lb, ub)
end

# ── Pretty Printing ──────────────────────────────────────────────────────────

function Base.show(io::IO, r::EventResult)
    print(io, "EventResult(τ=$(r.time), event=$(r.event_type), " *
              "state=$(r.state), params=(α=$(r.params.α), β=$(r.params.β), " *
              "δ=$(r.params.δ), γ=$(r.params.γ)))")
end

function Base.show(io::IO, r::OptimizationResult)
    println(io, "OptimizationResult(method=$(r.method))")
    println(io, "  Best: τ=$(r.best.time), event=$(r.best.event_type)")
    println(io, "  Params: α=$(r.best.params.α), β=$(r.best.params.β), " *
                "δ=$(r.best.params.δ), γ=$(r.best.params.γ)")
    print(io, "  Wall time: $(r.wall_time)s")
end
