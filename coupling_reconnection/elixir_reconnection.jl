using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

# Trixi does not yet implement shock capturing (VolumeIntegralShockCapturingHG) for
# StructuredMeshView. The cache layout and computation are identical to StructuredMesh{2},
# so we forward the missing dispatch methods to the parent StructuredMesh{2}.
#
# Note: calc_normalvectors_subcell_fv! and NormalVectorContainer2D dispatch on the mesh
# type only for method selection; all actual data access goes through cache_containers,
# which is specific to the StructuredMeshView. Forwarding to mesh.parent is therefore safe.
import Trixi: create_cache, fv_kernel!, flux, max_abs_speed_naive,
              flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal,
              flux_nonconservative_central, AbstractVolumeIntegralSubcell

# ── StructuredMeshView forwarding ──────────────────────────────────────────────
# Trixi's create_cache for VolumeIntegralShockCapturingHG dispatches on
# Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}, T8codeMesh{2}}
# but is missing StructuredMeshView{2}. Forward to the parent StructuredMesh{2}.
function Trixi.create_cache(mesh::StructuredMeshView{2}, equations,
                             volume_integral::AbstractVolumeIntegralSubcell,
                             dg, cache_containers, uEltype)
    Trixi.create_cache(mesh.parent, equations, volume_integral, dg, cache_containers,
                       uEltype)
end

# Same omission for fv_kernel! — calc_volume_integral! passes typeof(mesh), so
# dispatch on ::Type{<:StructuredMeshView{2}} and forward to StructuredMesh{2}.
@inline function Trixi.fv_kernel!(du, u,
                                  ::Type{<:StructuredMeshView{2}},
                                  have_nonconservative_terms, equations,
                                  volume_flux_fv, dg::DGSEM, cache, element, alpha = true)
    Trixi.fv_kernel!(du, u, StructuredMesh{2},
                     have_nonconservative_terms, equations,
                     volume_flux_fv, dg, cache, element, alpha)
end

# ── Normal-vector methods for IdealGlmMhdMultiIonEquations2D ──────────────────
# The StructuredMesh flux_differencing_kernel! passes contravariant vectors
# (AbstractVector) as the orientation argument, but the 2D multi-ion equations
# only have Integer-orientation methods. For our uniform Cartesian mesh the
# contravariant vectors are axis-aligned, so the linear combination below is
# exact (f(u, n) = n₁·f(u, 1) + n₂·f(u, 2)).

@inline function flux(u, normal_direction::AbstractVector,
                      equations::IdealGlmMhdMultiIonEquations2D)
    return (normal_direction[1] * flux(u, 1, equations) +
            normal_direction[2] * flux(u, 2, equations))
end

# LLF uses max_abs_speed_naive(u_ll, u_rr, normal_direction, equations) directly
# (without prior normalization in FluxPlusDissipation). For axis-aligned normals
# n = (h,0) or (0,h) this reduces to h·λ_x or h·λ_y respectively.
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::IdealGlmMhdMultiIonEquations2D)
    return (abs(normal_direction[1]) * max_abs_speed_naive(u_ll, u_rr, 1, equations) +
            abs(normal_direction[2]) * max_abs_speed_naive(u_ll, u_rr, 2, equations))
end

@inline function flux_ruedaramirez_etal(u_ll, u_rr, normal_direction::AbstractVector,
                                        equations::IdealGlmMhdMultiIonEquations2D)
    return (normal_direction[1] * flux_ruedaramirez_etal(u_ll, u_rr, 1, equations) +
            normal_direction[2] * flux_ruedaramirez_etal(u_ll, u_rr, 2, equations))
end

@inline function flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                                        normal_direction::AbstractVector,
                                                        equations::IdealGlmMhdMultiIonEquations2D)
    return (normal_direction[1] *
            flux_nonconservative_ruedaramirez_etal(u_ll, u_rr, 1, equations) +
            normal_direction[2] *
            flux_nonconservative_ruedaramirez_etal(u_ll, u_rr, 2, equations))
end

@inline function flux_nonconservative_central(u_ll, u_rr, normal_direction::AbstractVector,
                                              equations::IdealGlmMhdMultiIonEquations2D)
    return (normal_direction[1] *
            flux_nonconservative_central(u_ll, u_rr, 1, equations) +
            normal_direction[2] *
            flux_nonconservative_central(u_ll, u_rr, 2, equations))
end

"""
Adaptive coupling between a multi-ion MHD system and 2 MHD systems.
"""

# ── Absorbing ψ boundary condition ────────────────────────────────────────────
# BoundaryConditionDirichlet sets ψ=0 at physical boundaries (from initial_condition),
# which creates a hard reflecting wall for GLM divergence-cleaning waves.  GLM waves
# generated near the reconnection X-point propagate outward, hit this wall, and
# reflect back into the domain, building up ψ ripples that eventually destabilise
# the simulation.
#
# BoundaryConditionAbsorbingPsi is identical to BoundaryConditionDirichlet except
# that ψ (always the LAST variable in both GLM equation systems) is taken from the
# interior state rather than the reference state.  This is a zero-gradient (outflow)
# BC for ψ, which lets GLM waves pass through without reflection.
struct BoundaryConditionAbsorbingPsi{F}
    boundary_value_function::F
end

@inline function (bc::BoundaryConditionAbsorbingPsi)(u_inner, orientation_or_normal,
                                                      direction, x, t,
                                                      surface_flux_function, equations)
    u_ref = bc.boundary_value_function(x, t, equations)
    n = length(u_inner)
    # Replace last variable (ψ) with interior value; keep all other reference values.
    u_boundary = SVector(ntuple(Val(n)) do i
        i < n ? u_ref[i] : u_inner[n]
    end)
    if iseven(direction) # u_inner is "left", u_boundary is "right"
        return surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
    else # u_boundary is "left", u_inner is "right"
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
        flux = surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
        noncons_flux = nonconservative_flux_function(u_inner, u_boundary,
                                                     orientation_or_normal, equations)
    else
        flux = surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
        noncons_flux = nonconservative_flux_function(u_inner, u_boundary,
                                                     orientation_or_normal, equations)
    end
    return flux, noncons_flux
end


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

    v1 = 1e-2 * sin(2*pi*x[1]) * cos(pi*x[2])
    v2 = 1e-2 * cos(pi*x[1]) * sin(2*pi*x[2])
    v3 = 0.0

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

    v11 = 1e-2 * sin(2*pi*x[1]) * cos(pi*x[2])
    v12 = 1e-2 * cos(pi*x[1]) * sin(2*pi*x[2])
    v21 = v11
    v22 = v12
    v13 = 0.0
    v23 = 0.0

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
equations_mhd = IdealGlmMhdEquations2D(5/3, initial_c_h = 1.0)
equations_mion = IdealGlmMhdMultiIonEquations2D(gammas = (5 / 3, 5 / 3),
                                                charge_to_mass = (25.0,
                                                                  25.0), # [nondimensional] reduced from 76.3 → d_i=1/25=0.04 > Δy=0.01 (4 cells/d_i)
                                                gas_constants = (1.0, 1.0), # [nondimensional]
                                                molar_masses = (1.0, 1.0), # [nondimensional]
                                                ion_ion_collision_constants = [0.0 0.4079382480442680;
                                                                               0.4079382480442680 0.0], # [nondimensional] (computed with eq (4.142) of Schunk & Nagy (2009))
                                                ion_electron_collision_constants = (8.56368379833E-06,
                                                                                 8.56368379833E-06), # [nondimensional] (computed with eq (9) of Ghosh et al. (2019))
                                                electron_pressure = electron_pressure_constantTe,
                                                electron_temperature = electron_temperature_constantTe,
                                                initial_c_h = 1.0) # Initial GLM speed; updated each step by GlmSpeedCallback

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
# Resolution note: the ion skin depth d_i = 1/charge_to_mass ≈ 1/76.3 ≈ 0.013.
# The Hall physics that regularizes the X-point singularity requires Δy < d_i.
# With 100 cells in y (Δy = 0.01) the mion domain (30 cells) achieves Δy < d_i,
# allowing the Hall diffusion region to form instead of driving a singularity.
cells_per_dimension_parent = (100, 200)
coordinates_min = (-0.5, -0.5)
coordinates_max = (0.5, 0.5)
parent_mesh = StructuredMesh(cells_per_dimension_parent, coordinates_min, coordinates_max, periodicity=(false, false))

# Setup up the mesh views.
# Physical y-extents: bottom [-0.5, -0.15], middle [-0.15, 0.15], top [0.15, 0.5]
# (mion domain is widened from ±0.1 to ±0.15 to give 60 cells at Δy=0.005)
mesh_bottom = StructuredMeshView(parent_mesh;
                                 indices_min = (1, 1),
                                 indices_max = (100, 70))
mesh_middle = StructuredMeshView(parent_mesh;
                                 indices_min = (1, 71),
                                 indices_max = (100, 130))
mesh_top = StructuredMeshView(parent_mesh;
                              indices_min = (1, 131),
                              indices_max = (100, 200))

# Define the coupling functions.
#
# Energy convention difference between the two equation systems:
#   IdealGlmMhdEquations2D:          E = ρ|v|²/2 + p/(γ-1) + (B²+ψ²)/2   (includes B and ψ energy)
#   IdealGlmMhdMultiIonEquations2D:  Eₖ = ρₖ|vₖ|²/2 + pₖ/(γₖ-1)          (kinetic+internal only)
# The magnetic energy B²/2 is shared in the multi-ion system (not per species).
# ψ is a purely numerical GLM auxiliary variable — NOT physical magnetic energy.
# When crossing the interface we add/subtract B²/2 only (NOT ψ²/2).
#
# u (multi-ion conservative): [B1, B2, B3, ρ₁, ρ₁v₁₁, ρ₁v₁₂, ρ₁v₁₃, E₁,
#                               ρ₂, ρ₂v₂₁, ρ₂v₂₂, ρ₂v₂₃, E₂, ψ]
coupling_function_mion_mhd = (x, u, equations_other, equations_own) -> begin
    rho1 = max(u[4], eps(eltype(u)))
    rho2 = max(u[9], eps(eltype(u)))
    rho_total = rho1 + rho2
    mv1 = u[5] + u[10]
    mv2 = u[6] + u[11]
    mv3 = u[7] + u[12]
    # Individual species KEs (needed to extract thermal pressure from each E_k)
    KE1 = (u[5]^2 + u[6]^2 + u[7]^2) / (2 * rho1)
    KE2 = (u[10]^2 + u[11]^2 + u[12]^2) / (2 * rho2)
    # Centre-of-mass KE using total momentum (< KE1+KE2 when species velocities differ)
    KE_CM = (mv1^2 + mv2^2 + mv3^2) / (2 * rho_total)
    # E_MHD = p_total/(γ-1) + KE_CM + B²/2
    # p_total/(γ-1) = (E₁-KE₁) + (E₂-KE₂)  [mion Eₖ = KE_k + thermal_k]
    E_MHD = (u[8] - KE1) + (u[13] - KE2) + KE_CM + (u[1]^2 + u[2]^2 + u[3]^2)/2
    SVector(rho_total, mv1, mv2, mv3, E_MHD, u[1], u[2], u[3], zero(eltype(u)))
end
# u (MHD conservative): [ρ, ρv₁, ρv₂, ρv₃, E, B1, B2, B3, ψ]
coupling_function_mhd_mion = (x, u, equations_other, equations_own) -> begin
    # Strip B²/2 from MHD energy to get the non-magnetic energy for each species.
    # ψ²/2 is NOT subtracted: ψ is a numerical GLM variable, not physical magnetic energy.
    B_sq_half = (u[6]^2 + u[7]^2 + u[8]^2) / 2
    # Clamp to kinetic energy to guarantee p ≥ 0 if the entropy-stable scheme
    # has produced a slightly negative pressure near the interface.
    KE = (u[2]^2 + u[3]^2 + u[4]^2) / (2 * max(u[1], eps(Float64)))
    E_nonmag = max(u[5] - B_sq_half, KE)
    SVector(u[6], u[7], u[8],
            u[1]/2, u[2]/2, u[3]/2, u[4]/2, E_nonmag/2,  # species 1: half of total
            u[1]/2, u[2]/2, u[3]/2, u[4]/2, E_nonmag/2,  # species 2: half of total
            zero(eltype(u)))                                # ψ = 0: absorbing at interface
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

# ── Dedner parabolic GLM damping ──────────────────────────────────────────────
# Plain hyperbolic GLM transports ∇·B errors to the boundary at speed c_h.
# Adding the parabolic term ∂ψ/∂t = -ψ/τ (Dedner 2002) also damps ψ in the
# interior, making cleaning effective even far from the outflow boundaries.
# τ = L/c_h: ψ decays by 1/e in the time a GLM wave crosses the domain (L=0.5).
# ψ is always the last variable in both GLM equation systems.
function source_terms_glm_damping(u, x, t, equations::IdealGlmMhdEquations2D)
    L = 0.5f0
    nv = nvariables(equations)
    rate = equations.c_h / L
    return SVector(ntuple(Val(nv)) do i
        i == nv ? -rate * u[i] : zero(eltype(u))
    end)
end

function source_terms_glm_damping(u, x, t, equations::IdealGlmMhdMultiIonEquations2D)
    L = 0.5f0
    nv = nvariables(equations)
    rate = equations.c_h / L
    return SVector(ntuple(Val(nv)) do i
        i == nv ? -rate * u[i] : zero(eltype(u))
    end)
end

# Define the semidisretizations.
solver_bottom = DGSEM(basis_mhd, surface_flux_mhd, volume_integral_mhd)
boundary_conditions_bottom = (x_neg=BoundaryConditionAbsorbingPsi(initial_condition_mhd),
                              x_pos=BoundaryConditionAbsorbingPsi(initial_condition_mhd),
                              y_neg=BoundaryConditionAbsorbingPsi(initial_condition_mhd),
                              y_pos=BoundaryConditionCoupled(2, (:i_forward, :begin), Float64, coupling_function_mion_mhd),)
semi_bottom = SemidiscretizationHyperbolic(mesh_bottom, equations_mhd,
                                           initial_condition_mhd, solver_bottom,
                                           boundary_conditions=boundary_conditions_bottom,
                                           source_terms=source_terms_glm_damping)

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
boundary_conditions_middle = (x_neg=BoundaryConditionAbsorbingPsi(initial_condition_mionmhd),
                              x_pos=BoundaryConditionAbsorbingPsi(initial_condition_mionmhd),
                              y_neg=BoundaryConditionCoupled(1, (:i_forward, :end), Float64, coupling_function_mhd_mion),
                              y_pos=BoundaryConditionCoupled(3, (:i_forward, :begin), Float64, coupling_function_mhd_mion),)
semi_middle = SemidiscretizationHyperbolic(mesh_middle, equations_mion,
                                           initial_condition_mionmhd, solver_middle,
                                           boundary_conditions=boundary_conditions_middle,
                                           source_terms=source_terms_glm_damping)

solver_top = DGSEM(basis_mhd, surface_flux_mhd, volume_integral_mhd)
boundary_conditions_top = (; x_neg=BoundaryConditionAbsorbingPsi(initial_condition_mhd),
                           x_pos=BoundaryConditionAbsorbingPsi(initial_condition_mhd),
                           y_neg=BoundaryConditionCoupled(2, (:i_forward, :end), Float64, coupling_function_mion_mhd),
                           y_pos=BoundaryConditionAbsorbingPsi(initial_condition_mhd),)
semi_top = SemidiscretizationHyperbolic(mesh_top, equations_mhd,
                                        initial_condition_mhd, solver_top,
                                        boundary_conditions=boundary_conditions_top,
                                        source_terms=source_terms_glm_damping)

# coupled semidiscretization.
semi = SemidiscretizationCoupled(semi_bottom, semi_middle, semi_top)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0) # 100 [ps]

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# Define the CFL condition.
cfl = 0.1

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
glm_speed_callback = GlmSpeedCallback(glm_scale=0.9, cfl=cfl, semi_indices=[1, 2, 3]) # all three domains use GLM

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

    # Minimum thermal pressure per ion species.  At p_min ≈ 5% of initial minimum
    # pressure the limiter activates well before the catastrophic collapse at t≈0.32,
    # giving the scheme multiple steps of correction margin.  A value of 1e-4 is too
    # small: the 2N first-stage has no limiter call, so pressure can cross zero in
    # that un-guarded first sub-step.
    p_min = 1e-3

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
