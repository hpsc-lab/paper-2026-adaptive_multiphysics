using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Chromosphere–corona coupled simulation
# Bottom domain: CompressibleEulerEquations2D (chromosphere, neutral gas, no B)
# Top domain:    IdealGlmMhdEquations2D       (corona, vertical guide field B₂=1)
#
# A Gaussian pressure pulse at (0, -0.25) seeds an upward acoustic wave in the
# chromosphere.  At the neutral/magnetised interface (y=0) the wave mode-converts
# to fast-magnetosonic and Alfvén waves in the corona.
#
# Physical parameters (normalised units):
#   Chromosphere: ρ=1, p=1  →  c_s = √(γ p/ρ) = √(5/3) ≈ 1.29
#   Corona:       ρ=1, p=1, B₂=1  →  v_A = 1.0,  c_fast = √(c_s²+v_A²) ≈ 1.63
#   Equal densities/pressures across the interface eliminate equilibrium flux.
###############################################################################

function initial_condition_euler(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    v1  = 0.0
    v2  = 0.0
    p   = 1.0 + 0.01 * exp(-(x[1]^2 + (x[2] + 0.25)^2) / (2 * 0.05^2))
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

function initial_condition_mhd(x, t, equations::IdealGlmMhdEquations2D)
    rho = 1.0
    v1 = 0.0; v2 = 0.0; v3 = 0.0
    p   = 1.0
    B1  = 0.0; B2 = 0.0; B3 = 0.0
    psi = 0.0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

###############################################################################
# Coupling functions
#
# Euler conservative: [ρ, ρv₁, ρv₂, E_euler]
#   E_euler = ρ(v₁²+v₂²)/2 + p/(γ-1)
# MHD  conservative: [ρ, ρv₁, ρv₂, ρv₃, E_mhd, B₁, B₂, B₃, ψ]
#   E_mhd = ρ|v|²/2 + p/(γ-1) + |B|²/2
#
# Equilibrium check (v=0, ρ=p=1, B₂=1):
#   Euler→MHD ghost:  E = 1.5 + 0.5 = 2.0  ✓ matches MHD background
#   MHD→Euler ghost:  E = 2.0 - 0.5 = 1.5  ✓ matches Euler background
###############################################################################

# Euler state → MHD ghost: complete with background vertical field B₂=1
coupling_euler_to_mhd = (x, u, equations_other, equations_own) -> begin
    T  = eltype(u)
    B2 = one(T)  # corona background field
    SVector(u[1], u[2], u[3], zero(T), u[4] + B2^2 / 2, zero(T), B2, zero(T), zero(T))
end

# MHD state → Euler ghost: remove magnetic energy and out-of-plane kinetic energy
coupling_mhd_to_euler = (x, u, equations_other, equations_own) -> begin
    B_sq_half = (u[6]^2 + u[7]^2 + u[8]^2) / 2
    v3_KE     = u[4]^2 / (2 * max(u[1], eps(eltype(u))))
    SVector(u[1], u[2], u[3], u[5] - B_sq_half - v3_KE)
end

###############################################################################
# Absorbing ψ boundary condition
# Passes interior ψ through as the ghost value instead of forcing ψ=0.
# This lets GLM waves exit at y_pos without reflecting back.
###############################################################################

struct BoundaryConditionAbsorbingPsi{F}
    boundary_value_function::F
end

@inline function (bc::BoundaryConditionAbsorbingPsi)(u_inner, orientation_or_normal,
                                                      direction, x, t,
                                                      surface_flux_function, equations)
    u_ref = bc.boundary_value_function(x, t, equations)
    n = length(u_inner)
    u_boundary = SVector(ntuple(Val(n)) do i
        i < n ? u_ref[i] : u_inner[n]
    end)
    if iseven(direction)
        return surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
    else
        return surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
    end
end

@inline function (bc::BoundaryConditionAbsorbingPsi)(u_inner, orientation_or_normal,
                                                      direction, x, t,
                                                      surface_flux_functions::Tuple,
                                                      equations)
    surface_flux_function, nonconservative_flux_function = surface_flux_functions
    u_ref = bc.boundary_value_function(x, t, equations)
    n = length(u_inner)
    u_boundary = SVector(ntuple(Val(n)) do i
        i < n ? u_ref[i] : u_inner[n]
    end)
    if iseven(direction)
        flux_val     = surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
        noncons_flux = nonconservative_flux_function(u_inner, u_boundary,
                                                     orientation_or_normal, equations)
    else
        flux_val     = surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
        noncons_flux = nonconservative_flux_function(u_inner, u_boundary,
                                                     orientation_or_normal, equations)
    end
    return flux_val, noncons_flux
end

###############################################################################
# Dedner parabolic GLM damping: ∂ψ/∂t += -c_h/L · ψ  (Dedner 2002)
###############################################################################

function source_terms_glm_damping(u, x, t, equations::IdealGlmMhdEquations2D)
    L    = 0.5f0
    nv   = nvariables(equations)
    rate = equations.c_h / L
    return SVector(ntuple(Val(nv)) do i
        i == nv ? -rate * u[i] : zero(eltype(u))
    end)
end

###############################################################################
# Mesh: 100×100 parent, split 50/50 at y=0
###############################################################################

cells_per_dimension_parent = (100, 100)
coordinates_min = (-0.5, -0.5)
coordinates_max = ( 0.5,  0.5)
parent_mesh = StructuredMesh(cells_per_dimension_parent, coordinates_min, coordinates_max,
                             periodicity = (false, false))

mesh_bottom = StructuredMeshView(parent_mesh; indices_min = (1,  1), indices_max = (100,  50))
mesh_top    = StructuredMeshView(parent_mesh; indices_min = (1, 51), indices_max = (100, 100))

###############################################################################
# Equations and solvers
###############################################################################

equations_euler = CompressibleEulerEquations2D(5 / 3)
equations_mhd   = IdealGlmMhdEquations2D(5 / 3, initial_c_h = 1.0)

solver_euler = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
                     volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

solver_mhd = DGSEM(polydeg = 3,
                   surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                   volume_integral = VolumeIntegralFluxDifferencing(
                       (flux_hindenlang_gassner, flux_nonconservative_powell)))

###############################################################################
# Semidiscretizations
###############################################################################

# Bottom (Euler / chromosphere): Dirichlet on all physical boundaries, coupled at y_pos
boundary_conditions_bottom = (
    x_neg = BoundaryConditionDirichlet(initial_condition_euler),
    x_pos = BoundaryConditionDirichlet(initial_condition_euler),
    y_neg = BoundaryConditionDirichlet(initial_condition_euler),
    y_pos = BoundaryConditionCoupled(2, (:i_forward, :begin), Float64,
                                     coupling_mhd_to_euler),
)

semi_bottom = SemidiscretizationHyperbolic(mesh_bottom, equations_euler,
                                           initial_condition_euler, solver_euler,
                                           boundary_conditions = boundary_conditions_bottom)

# Top (MHD / corona): absorbing on physical boundaries, coupled at y_neg
boundary_conditions_top = (
    x_neg = BoundaryConditionAbsorbingPsi(initial_condition_mhd),
    x_pos = BoundaryConditionAbsorbingPsi(initial_condition_mhd),
    y_neg = BoundaryConditionCoupled(1, (:i_forward, :end), Float64,
                                     coupling_euler_to_mhd),
    y_pos = BoundaryConditionAbsorbingPsi(initial_condition_mhd),
)

semi_top = SemidiscretizationHyperbolic(mesh_top, equations_mhd,
                                        initial_condition_mhd, solver_mhd,
                                        boundary_conditions = boundary_conditions_top,
                                        source_terms = source_terms_glm_damping)

semi = SemidiscretizationCoupled(semi_bottom, semi_top)

###############################################################################
# ODE solvers, callbacks
###############################################################################

tspan = (0.0, 1.0)
ode   = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
alive_callback    = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution   = true,
                                     output_directory      = "out",
                                     solution_variables    = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

# GLM speed update only for the MHD (corona) domain
glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = 0.5, semi_indices = [2])

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0,
            ode_default_options()..., callback = callbacks)

summary_callback()
