module LVMaxTime_lite

# Data structures and small utility functions
include("types.jl")

# ODE integration with event-surface detection
include("ode_solver.jl")

# Two-stage parameter optimization (global + local)
include("optimize.jl")

# Rigorous Taylor-model flowpipe certification
include("certify.jl")

# ── Exports ──────────────────────────────────────────────────────────────────

# Types
export LVParams, LVProblemSpec, SolverConfig
export EventType, EVT_DX1_ZERO, EVT_DX2_ZERO, EVT_DETJ_ZERO, EVT_NONE
export EventResult, OptimizationResult
export CertificationStatus, CERT_VERIFIED, CERT_SAFETY_ONLY, CERT_FAILED
export Certificate

# Utilities
export params_from_vector, params_to_vector, equilibrium, estimated_period
export rhs, jacobian_det, is_at_equilibrium, is_on_event_surface, bounds_to_ranges

# Core pipeline
export first_event_time
export optimize_min_event_time
export certify_event_time, certify_optimum

end # module
