using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

# Trixi does not yet implement shock capturing (VolumeIntegralShockCapturingHG) for
# StructuredMeshView. The cache layout and computation are identical to StructuredMesh{2},
# so we forward the missing dispatch methods to the parent StructuredMesh{2}.
#
# Note: calc_normalvectors_subcell_fv! and NormalVectorContainer2D dispatch on the mesh
# type only for method selection; all actual data access goes through cache_containers,
# which is specific to the StructuredMeshView. Forwarding to mesh.parent is therefore safe.
import Trixi: create_cache, fv_kernel!

@inline function Trixi.fv_kernel!(du, u, ::Type{<:StructuredMeshView{2}},
                                  have_nonconservative_terms, equations,
                                  volume_flux_fv, dg, cache, element, alpha = true)
    Trixi.fv_kernel!(du, u, StructuredMesh{2},
                    have_nonconservative_terms, equations,
                    volume_flux_fv, dg, cache, element, alpha)
end

function Trixi.create_cache(mesh::StructuredMeshView{2}, equations,
                             volume_integral::Trixi.AbstractVolumeIntegralSubcell,
                             dg, cache_containers, uEltype)
    # Forward to StructuredMesh{2} — NormalVectorContainer2D and create_f_threaded
    # both read exclusively from cache_containers (not the mesh object), so using
    # mesh.parent for dispatch while keeping the view's cache_containers is correct.
    Trixi.create_cache(mesh.parent, equations, volume_integral, dg, cache_containers, uEltype)
end

"""
Adaptive coupling between a multi-ion MHD system and 2 MHD systems.
"""


"""
Define the initial condition for the MHD domain as two magnetic flux rings
that are being pushed against each other.
"""
function initial_condition_mhd(x, t, equations::IdealGlmMhdEquations2D)
    # https://arxiv.org/pdf/2510.01060
    delta = 0.1 # Current sheet thickness.
    Bz = 0.5 # Guiding magnetic field.
    beta = 1.0 # Magnetic beta.

    B1 = -cos(pi*x[1])*cos(2*pi*x[2])*tanh(x[2]/delta) -
        (1 - tanh(x[2]/delta)^2)*sin(2*pi*x[2])*cos(pi*x[1])/(2*pi*delta)
    B2 = -sin(pi*x[1])*sin(2*pi*x[2])*tanh(x[2]/delta)/2
    B3 = Bz

    p_mag = (B1^2 + B2^2 + B3^2)/2
    p_thermal = beta*p_mag
    p = p_thermal  # thermal pressure only; prim2cons adds B²/2 to energy internally

    rho = 1.0  # uniform background density; pressure balance is maintained via p_thermal

    # Add a velocity that pushes the magnetic field towards the center in y
    # and outwards in x.
    r = sqrt((x[1] - 1)^2 + (x[2] - 1)^2)
    v1 = -(x[2] - 1) * r * exp(-r^2*5)
    v2 = (x[1] - 1) * r * exp(-r^2*5)

    r = sqrt((x[1] + 1)^2 + (x[2] - 1)^2)
    v1 = v1 + (x[2] - 1) * r * exp(-r^2*5)
    v2 = v2 - (x[1] + 1) * r * exp(-r^2*5)

    r = sqrt((x[1] - 1)^2 + (x[2] + 1)^2)
    v1 = v1 + (x[2] + 1) * r * exp(-r^2*5)
    v2 = v2 - (x[1] - 1) * r * exp(-r^2*5)

    r = sqrt((x[1] + 1)^2 + (x[2] + 1)^2)
    v1 = v1 - (x[2] + 1) * r * exp(-r^2*5)
    v2 = v2 + (x[1] + 1) * r * exp(-r^2*5)

    v3 = 0.0

    # Add a small deterministic perturbation to the velocity field.
    v1 += 1e-3 * sin(2*pi*x[1]) * cos(pi*x[2])
    v2 += 1e-3 * cos(pi*x[1]) * sin(2*pi*x[2])

    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

function initial_condition_mionmhd(x, t, equations::IdealGlmMhdMultiIonEquations2D)
    # https://arxiv.org/pdf/2510.01060
    delta = 0.1 # Current sheet thickness.
    Bz = 0.5 # Guiding magnetic field.
    beta = 1.0 # Magnetic beta.

    B1 = -cos(pi*x[1])*cos(2*pi*x[2])*tanh(x[2]/delta) -
        (1 - tanh(x[2]/delta)^2)*sin(2*pi*x[2])*cos(pi*x[1])/(2*pi*delta)
    B2 = -sin(pi*x[1])*sin(2*pi*x[2])*tanh(x[2]/delta)/2
    B3 = Bz

    p_mag = (B1^2 + B2^2 + B3^2)/2
    p_thermal = beta*p_mag
    p1 = p_thermal/2  # split thermal pressure equally between two species
    p2 = p_thermal/2

    rho1 = 0.5  # uniform background density split equally between species
    rho2 = 0.5

    # Perturbation of the velocity.
    # Add a velocity that pushes the magnetic field towards the center in y
    # and outwards in x.
    r = sqrt((x[1] - 1)^2 + (x[2] - 1)^2)
    v11 = -(x[2] - 1) * r * exp(-r^2*5)
    v12 = (x[1] - 1) * r * exp(-r^2*5)

    r = sqrt((x[1] + 1)^2 + (x[2] - 1)^2)
    v11 = v11 + (x[2] - 1) * r * exp(-r^2*5)
    v12 = v12 - (x[1] + 1) * r * exp(-r^2*5)

    r = sqrt((x[1] - 1)^2 + (x[2] + 1)^2)
    v11 = v11 + (x[2] + 1) * r * exp(-r^2*5)
    v12 = v12 - (x[1] - 1) * r * exp(-r^2*5)

    r = sqrt((x[1] + 1)^2 + (x[2] + 1)^2)
    v11 = v11 - (x[2] + 1) * r * exp(-r^2*5)
    v12 = v12 + (x[1] + 1) * r * exp(-r^2*5)

    v21 = v11
    v22 = v12

    v13 = 0.0
    v23 = 0.0

    # Add a small deterministic perturbation to the velocity field.
    v11 += 1e-3 * sin(2*pi*x[1]) * cos(pi*x[2])
    v12 += 1e-3 * cos(pi*x[1]) * sin(2*pi*x[2])
    v21 += 1e-3 * sin(2*pi*x[1]) * cos(pi*x[2])
    v22 += 1e-3 * cos(pi*x[1]) * sin(2*pi*x[2])

    psi = 0.0

    return prim2cons(SVector(B1, B2, B3, rho1, v11, v12, v13, p1, rho2, v21, v22, v23, p2, psi),
                     equations)
end

# Return the electron pressure for a constant electron temperature Te = 1 keV
function electron_pressure_constantTe(u, equations::IdealGlmMhdMultiIonEquations2D)
    @unpack charge_to_mass = equations
    Te = electron_temperature_constantTe(u, equations)
    total_electron_charge = zero(eltype(u))
    for k in eachcomponent(equations)
        rho_k = u[3 + (k - 1) * 5 + 1]
        total_electron_charge += rho_k * charge_to_mass[k]
    end

    # Boltzmann constant divided by elementary charge
    RealT = eltype(u)
    kB_e = convert(RealT, 7.86319034E-02) #[nondimensional]

    return total_electron_charge * kB_e * Te
end

# Return the constant electron temperature Te = 1 keV
function electron_temperature_constantTe(u, equations::IdealGlmMhdMultiIonEquations2D)
    RealT = eltype(u)
    return convert(RealT, 0.008029953773) # [nondimensional] = 1 [keV]
end

# Define the two set of partial differential equations.
equations_mhd = IdealGlmMhdEquations2D(5/3)
equations_mion = IdealGlmMhdMultiIonEquations2D(gammas = (5 / 3, 5 / 3),
                                                charge_to_mass = (76.3049060157692000,
                                                                  76.3049060157692000), # [nondimensional]
                                                gas_constants = (1.0, 1.0), # [nondimensional]
                                                molar_masses = (1.0, 1.0), # [nondimensional]
                                                ion_ion_collision_constants = [0.0 0.4079382480442680;
                                                                               0.4079382480442680 0.0], # [nondimensional] (computed with eq (4.142) of Schunk & Nagy (2009))
                                                ion_electron_collision_constants = (8.56368379833E-06,
                                                                                 8.56368379833E-06), # [nondimensional] (computed with eq (9) of Ghosh et al. (2019))
                                                electron_pressure = electron_pressure_constantTe,
                                                electron_temperature = electron_temperature_constantTe,
                                                initial_c_h = 0.0) # Deactivate GLM divergence cleaning

# Temperature of ion 1
function temperature1(u, equations::IdealGlmMhdMultiIonEquations2D)
    rho_1, _ = Trixi.get_component(1, u, equations)
    p = pressure(u, equations)

    return p[1] / (rho_1 * equations.gas_constants[1])
end

# Temperature of ion 2
function temperature2(u, equations::IdealGlmMhdMultiIonEquations2D)
    rho_2, _ = Trixi.get_component(2, u, equations)
    p = pressure(u, equations)

    return p[2] / (rho_2 * equations.gas_constants[2])
end

# Set up the parent domain.
cells_per_dimension_parent = (50, 50)
coordinates_min = (-0.5, -0.5)
coordinates_max = (0.5, 0.5)
parent_mesh = StructuredMesh(cells_per_dimension_parent, coordinates_min, coordinates_max, periodicity=(false, false))

# Setup up the mesh views.
mesh_bottom = StructuredMeshView(parent_mesh;
                                 indices_min = (1, 1),
                                 indices_max = (50, 20))
mesh_middle = StructuredMeshView(parent_mesh;
                                 indices_min = (1, 21),
                                 indices_max = (50, 30))
mesh_top = StructuredMeshView(parent_mesh;
                              indices_min = (1, 31),
                              indices_max = (50, 50))

# Define the coupling functions.
#
# Energy convention difference between the two equation systems:
#   IdealGlmMhdEquations2D:          E = ρ|v|²/2 + p/(γ-1) + (B²+ψ²)/2   (includes magnetic energy)
#   IdealGlmMhdMultiIonEquations2D:  Eₖ = ρₖ|vₖ|²/2 + pₖ/(γₖ-1)          (kinetic+internal only)
# The magnetic energy (B²+ψ²)/2 is a shared field in the multi-ion system, not stored per species.
# The coupling functions must add/subtract (B²+ψ²)/2 when crossing the interface.
#
# u (multi-ion conservative): [B1, B2, B3, ρ₁, ρ₁v₁₁, ρ₁v₁₂, ρ₁v₁₃, E₁,
#                               ρ₂, ρ₂v₂₁, ρ₂v₂₂, ρ₂v₂₃, E₂, ψ]
coupling_function_mion_mhd = (x, u, equations_other, equations_own) -> SVector(
    u[4] + u[9],
    u[5] + u[10],
    u[6] + u[11],
    u[7] + u[12],
    u[8] + u[13] + (u[1]^2 + u[2]^2 + u[3]^2 + u[14]^2)/2,  # add shared magnetic energy
    u[1], u[2], u[3],
    u[14])
# u (MHD conservative): [ρ, ρv₁, ρv₂, ρv₃, E, B1, B2, B3, ψ]
coupling_function_mhd_mion = (x, u, equations_other, equations_own) -> begin
    B_sq_half = (u[6]^2 + u[7]^2 + u[8]^2 + u[9]^2) / 2
    # E_nonmag = ρv²/2 + p/(γ-1) ≥ 0 physically, but the entropy-stable MHD scheme
    # is NOT positivity-preserving: numerical oscillations can make B²/2 > E, giving
    # negative thermal energy. Clamp to kinetic energy (guarantees p ≥ 0 per species).
    KE = (u[2]^2 + u[3]^2 + u[4]^2) / (2 * max(u[1], eps(Float64)))
    E_nonmag = max(u[5] - B_sq_half, KE)
    SVector(u[6], u[7], u[8],
            u[1]/2, u[2]/2, u[3]/2, u[4]/2, E_nonmag/2,
            u[1]/2, u[2]/2, u[3]/2, u[4]/2, E_nonmag/2,
            u[9])
end
coupling_function_identity = (x, u, equations_other, equations_own) -> u

# Entropy conservative volume numerical fluxes with standard LLF dissipation at interfaces
volume_flux_mion = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux_mion = (flux_lax_friedrichs, flux_nonconservative_central)
volume_flux_mhd = (flux_hindenlang_gassner, flux_nonconservative_powell)
surface_flux_mhd = (flux_lax_friedrichs, flux_nonconservative_powell)

# Shock capturing for the MHD domains: the entropy-stable Hindenlang-Gassner flux
# is NOT positivity-preserving, so the MHD domains can develop negative pressure near
# the coupling boundary. Using shock capturing with pressure as indicator prevents this.
basis_mhd = LobattoLegendreBasis(3)
indicator_mhd = IndicatorHennemannGassner(equations_mhd, basis_mhd;
                                          alpha_max    = 1.0,
                                          alpha_min    = 0.001,
                                          alpha_smooth = false,
                                          variable     = pressure)
volume_integral_mhd = VolumeIntegralShockCapturingHG(indicator_mhd;
                                                     volume_flux_dg = volume_flux_mhd,
                                                     volume_flux_fv = surface_flux_mhd)

# Define the semidisretizations.
solver_bottom = DGSEM(basis_mhd, surface_flux_mhd, volume_integral_mhd)
boundary_conditions_bottom = (x_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                              x_pos=BoundaryConditionDirichlet(initial_condition_mhd),
                              y_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                              y_pos=BoundaryConditionCoupled(2, (:i_forward, :begin), Float64, coupling_function_mion_mhd),)
semi_bottom = SemidiscretizationHyperbolic(mesh_bottom, equations_mhd,
                                           initial_condition_mhd, solver_bottom,
                                           boundary_conditions=boundary_conditions_bottom)

# Shock capturing for the mion domain: the reconnection X-point creates a current
# singularity (p → 0, B → 0) that explicit ideal-MHD schemes cannot handle without
# local dissipation. VolumeIntegralShockCapturingHG detects under-resolved cells via
# modal energy decay (Hennemann & Gassner indicator) and blends in a first-order FV
# scheme locally, providing just enough dissipation to prevent negative pressure.
basis_middle = LobattoLegendreBasis(3)
# Use total thermal pressure (p1 + p2) as the indicator variable.
# At the reconnection X-point it is pressure (not density) that collapses to zero —
# density can peak in the current sheet — so pressure is a more reliable trigger.
# alpha_max = 1.0 allows full first-order FV fallback in cells where the indicator fires.
function mion_total_pressure(u, ::IdealGlmMhdMultiIonEquations2D)
    rho1 = max(u[4], eps(eltype(u))); E1 = u[8]
    rho2 = max(u[9], eps(eltype(u))); E2 = u[13]
    # p_k = (E_k - |ρv_k|²/(2ρ_k)) * (γ-1), γ = 5/3
    KE1 = (u[5]^2 + u[6]^2 + u[7]^2) / (2 * rho1)
    KE2 = (u[10]^2 + u[11]^2 + u[12]^2) / (2 * rho2)
    return (E1 - KE1 + E2 - KE2) * (2/3)  # (γ-1) = 2/3
end
indicator_middle = IndicatorHennemannGassner(equations_mion, basis_middle;
                                             alpha_max    = 1.0,
                                             alpha_min    = 0.001,
                                             alpha_smooth = false, # apply_smoothing! not implemented for StructuredMeshView
                                             variable     = mion_total_pressure)
volume_integral_middle = VolumeIntegralShockCapturingHG(indicator_middle;
                                                        volume_flux_dg = volume_flux_mion,
                                                        volume_flux_fv = surface_flux_mion)
solver_middle = DGSEM(basis_middle, surface_flux_mion, volume_integral_middle)
boundary_conditions_middle = (x_neg=BoundaryConditionDirichlet(initial_condition_mionmhd),
                              x_pos=BoundaryConditionDirichlet(initial_condition_mionmhd),
                              y_neg=BoundaryConditionCoupled(1, (:i_forward, :end), Float64, coupling_function_mhd_mion),
                              y_pos=BoundaryConditionCoupled(3, (:i_forward, :begin), Float64, coupling_function_mhd_mion),)
semi_middle = SemidiscretizationHyperbolic(mesh_middle, equations_mion,
                                           initial_condition_mionmhd, solver_middle,
                                           boundary_conditions=boundary_conditions_middle)

solver_top = DGSEM(basis_mhd, surface_flux_mhd, volume_integral_mhd)
boundary_conditions_top = (; x_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                           x_pos=BoundaryConditionDirichlet(initial_condition_mhd),
                           y_neg=BoundaryConditionCoupled(2, (:i_forward, :end), Float64, coupling_function_mion_mhd),
                           y_pos=BoundaryConditionDirichlet(initial_condition_mhd),)
semi_top = SemidiscretizationHyperbolic(mesh_top, equations_mhd,
                                        initial_condition_mhd, solver_top,
                                        boundary_conditions=boundary_conditions_top)

# coupled semidiscretization.
semi = SemidiscretizationCoupled(semi_bottom, semi_middle, semi_top)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0) # 100 [ps]

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# Define the CFL condition.
cfl = 0.01

analysis_interval = 10000
# analysis_callback_bottom = AnalysisCallback(semi_bottom, interval = 100)
# analysis_callback_middle = AnalysisCallback(semi_middle, interval = 100)
# analysis_callback_top = AnalysisCallback(semi_top, interval = 100)
# analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback_bottom,
#                                             analysis_callback_middle, analysis_callback_top)
# analysis_callback = AnalysisCallback(semi,
#                                      save_analysis = true,
#                                      interval = analysis_interval,
#                                      extra_analysis_integrals = (temperature1,
#                                                                  temperature2))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = cfl) # Very small CFL due to the stiff source terms

# The Generalized Lagrange Method divergence cleans the magnetic field.
glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl, semi_indices=[1, 3]) # semi 2 (mion) has GLM disabled

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory="out",
                                     solution_variables=cons2prim)

callbacks = CallbackSet(summary_callback,
#                         analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# Positivity-preserving stage limiter for the mion (middle) subdomain.
#
# At the reconnection X-point B→0 and p→0 simultaneously, so all wave speeds
# vanish. LLF-FV shock capturing provides zero dissipation there, and the
# Hennemann-Gassner indicator also gives zero alpha (smooth zero is not a shock).
# We therefore apply the Zhang-Shu positivity limiter after each RK stage to
# guarantee p1, p2 ≥ p_min throughout the mion domain.
#
# The limiter scales the cell mean toward the element mean to restore positivity
# without destroying conservation.
function mion_positivity_limiter!(u_ode, integrator, semi::SemidiscretizationCoupled, t)
    semi_mion = semi.semis[2]  # middle mion domain is semi index 2
    u_mion = Trixi.wrap_array(@view(u_ode[semi.u_indices[2]]), semi_mion)
    mesh_mion, equations_mion, solver_mion, cache_mion = Trixi.mesh_equations_solver_cache(semi_mion)

    # Minimum thermal pressure per ion species.  Chosen well above machine epsilon
    # but small enough to be physically neutral (≈ 0.5% of initial minimum pressure).
    p_min = 1e-4

    # Conservative pressure extractors (u is conservative state at one node):
    #   mion layout: [B1,B2,B3, ρ₁,ρ₁v₁₁,ρ₁v₁₂,ρ₁v₁₃,E₁, ρ₂,ρ₂v₂₁,ρ₂v₂₂,ρ₂v₂₃,E₂, ψ]
    #   p_k = (E_k - |ρ_k v_k|² / (2ρ_k)) * (γ_k - 1)
    function p1_cons(u, eq)
        rho1 = max(u[4], eps(eltype(u)))
        (u[8] - (u[5]^2 + u[6]^2 + u[7]^2) / (2 * rho1)) * (eq.gammas[1] - 1)
    end
    function p2_cons(u, eq)
        rho2 = max(u[9], eps(eltype(u)))
        (u[13] - (u[10]^2 + u[11]^2 + u[12]^2) / (2 * rho2)) * (eq.gammas[2] - 1)
    end

    Trixi.limiter_zhang_shu!(u_mion, p_min, p1_cons,
                             mesh_mion, equations_mion, solver_mion, cache_mion)
    Trixi.limiter_zhang_shu!(u_mion, p_min, p2_cons,
                             mesh_mion, equations_mion, solver_mion, cache_mion)
    return nothing
end

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false,
                                      stage_limiter! = mion_positivity_limiter!);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
