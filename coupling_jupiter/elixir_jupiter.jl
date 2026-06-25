using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using Random

###############################################################################
# Jupiter dynamo-boundary coupled simulation
# Bottom domain: IdealGlmMhdEquations2D    (metallic hydrogen interior, conducting)
# Top domain:    CompressibleEulerEquations2D (molecular hydrogen envelope, neutral)
#
# At ~0.85 R_J the hydrogen pressure-ionises and becomes electrically conducting.
# Below this "dynamo boundary" the fluid couples to the magnetic field (MHD);
# above it the gas is neutral and magnetic effects vanish (Euler).
#
# A multi-mode turbulent body force continuously stirs the molecular envelope,
# mimicking solar irradiation and large-scale convective overturning.  The driven
# acoustic and vortical motions reach the dynamo boundary and partially couple
# into fast-magnetosonic and Alfvén waves in the conducting interior.  The MHD
# domain starts at rest; all wave activity originates in the neutral layer above.
#
# Physical parameters (normalised units):
#   Metallic H (MHD):    ρ=5, p=1, γ=5/3, c_s = √(γp/ρ) = √(1/3) ≈ 0.577
#                        B field: weak seed field (B₀=1e-8), divergence-free,
#                        linear profile satisfying the perfect-conductor BC at y=-0.5:
#                          B_y = 2B₀(y+0.5)  →  0 at wall, B₀ at interface
#                          B_x = -2B₀·x      →  exactly cancels ∂B_y/∂y
#   Molecular H (Euler): ρ=1, p=1, γ=7/5, c_s = √(γp/ρ) = √(1.4) ≈ 1.18
#
# Density ratio ρ_metallic/ρ_molecular ≈ 5, consistent with interior models
# (Guillot 1999; Militzer et al. 2022) at the ~0.85 R_J dynamo boundary.
# Both layers start at p=1 with v=0, so the equilibrium flux is zero despite
# the density jump (flux depends on p and v, not ρ, at rest).
# The seed B field is so weak (B₀=1e-8) that |B|²/2 < machine-ε relative to
# the O(1) energy, so the equilibrium energy balance is unaffected.
###############################################################################

function initial_condition_mhd(x, t, equations::IdealGlmMhdEquations2D)
    rho = 5.0
    v1  = 0.0; v2 = 0.0; v3 = 0.0
    p   = 1.0
    # Divergence-free seed field compatible with the perfect-conductor wall at y = -0.5.
    # Linear B_y profile: 0 at the wall, B₀ at the coupling interface (y = 0).
    # div(B) = ∂B_x/∂x + ∂B_y/∂y = -2B₀ + 2B₀ = 0  exactly.
    B0  = 1e-8
    B1  = -2 * B0 * x[1]           # compensates ∂B_y/∂y to enforce div B = 0
    B2  =  2 * B0 * (x[2] + 0.5)   # 0 at y=-0.5, B₀ at y=0
    B3  = 0.0
    psi = 0.0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

function initial_condition_euler(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    v1  = 0.0
    v2  = 0.0
    p   = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

###############################################################################
# Coupling functions
#
# Rule: each domain's coupled BC receives the OTHER domain's state (u) and must
# return a ghost state in ITS OWN variables.
#
# Euler conservative: [ρ, ρv₁, ρv₂, E_euler]          γ_euler = 7/5
# MHD  conservative: [ρ, ρv₁, ρv₂, ρv₃, E_mhd, B₁, B₂, B₃, ψ]  γ_mhd = 5/3
#
# The interface is a contact discontinuity: ρ_mhd=5, ρ_euler=1 with p continuous.
# Naively passing the sender's density into the ghost creates a factor-of-5 density
# mismatch that the Lax-Friedrichs dissipation sees as a constant mass source,
# draining ρ from the MHD layer near the interface until ρ→0 and the run crashes.
#
# Fix: each ghost uses the RECEIVING domain's equilibrium density, with velocity
# and pressure extracted from the SENDER using its own γ.  This eliminates the
# spurious LF mass flux while correctly transmitting pressure perturbations.
#
# Equilibrium consistency check (ρ_mhd=5, p=1, v=0):
#   coupling_euler_to_mhd ghost: ρ=5, p=(0.4×2.5)=1, E=1/(2/3)=1.5  ✓
#   coupling_mhd_to_euler ghost: ρ=1, p=(2/3×1.5)=1, E=1/0.4=2.5    ✓
###############################################################################

# Background densities — must match the initial conditions.
const rho_mhd_eq   = 5.0   # metallic H equilibrium density
const rho_euler_eq = 1.0   # molecular H equilibrium density

# Euler state (γ=7/5) → MHD ghost (γ=5/3):
# Extract v and p from Euler; rebuild the ghost at MHD equilibrium density.
coupling_euler_to_mhd = (x, u, equations_other, equations_own) -> begin
    T       = eltype(u)
    inv_rho = one(T) / max(u[1], eps(T))
    v1      = u[2] * inv_rho
    v2      = u[3] * inv_rho
    KE      = (u[2]^2 + u[3]^2) * inv_rho / 2
    p       = (equations_other.gamma - 1) * (u[4] - KE)
    rho_g   = convert(T, rho_mhd_eq)
    B0      = convert(T, 1e-8)
    B1      = -2 * B0 * convert(T, x[1])
    B2      = B0
    E_mhd   = rho_g * (v1^2 + v2^2) / 2 + p / (equations_own.gamma - 1) +
              (B1^2 + B2^2) / 2
    SVector(rho_g, rho_g*v1, rho_g*v2, zero(T), E_mhd, B1, B2, zero(T), zero(T))
end

# MHD state (γ=5/3) → Euler ghost (γ=7/5):
# Extract v and p from MHD; rebuild the ghost at Euler equilibrium density.
coupling_mhd_to_euler = (x, u, equations_other, equations_own) -> begin
    T         = eltype(u)
    inv_rho   = one(T) / max(u[1], eps(T))
    v1        = u[2] * inv_rho
    v2        = u[3] * inv_rho
    v3        = u[4] * inv_rho
    KE_mhd    = (u[2]^2 + u[3]^2 + u[4]^2) * inv_rho / 2
    B_sq_half = (u[6]^2 + u[7]^2 + u[8]^2) / 2
    p         = (equations_other.gamma - 1) * (u[5] - KE_mhd - B_sq_half)
    rho_g     = convert(T, rho_euler_eq)
    E_euler   = rho_g * (v1^2 + v2^2) / 2 + p / (equations_own.gamma - 1)
    SVector(rho_g, rho_g*v1, rho_g*v2, E_euler)
end

###############################################################################
# Slip-wall boundary condition for IdealGlmMhdEquations2D.
# Ghost state mirrors the wall-normal momentum (ρvₙ → -ρvₙ) and normal field
# (Bₙ → -Bₙ), enforcing vₙ=0 and Bₙ=0 at the wall.  ψ is also negated so
# GLM divergence-cleaning waves reflect rather than accumulate at the wall.
###############################################################################

struct BoundaryConditionSlipWallMHD end

@inline function _mhd_slip_wall_ghost(u, orientation)
    # MHD conservative: [ρ, ρv₁, ρv₂, ρv₃, E, B₁, B₂, B₃, ψ]
    if orientation == 1  # x-normal wall: negate ρv₁, B₁, ψ
        SVector(u[1], -u[2], u[3], u[4], u[5], -u[6], u[7], u[8], -u[9])
    else                 # y-normal wall: negate ρv₂, B₂, ψ
        SVector(u[1], u[2], -u[3], u[4], u[5], u[6], -u[7], u[8], -u[9])
    end
end

@inline function (::BoundaryConditionSlipWallMHD)(u_inner, orientation_or_normal,
                                                   direction, x, t,
                                                   surface_flux_function,
                                                   equations::IdealGlmMhdEquations2D)
    u_ghost = _mhd_slip_wall_ghost(u_inner, orientation_or_normal)
    if iseven(direction)
        return surface_flux_function(u_inner, u_ghost, orientation_or_normal, equations)
    else
        return surface_flux_function(u_ghost, u_inner, orientation_or_normal, equations)
    end
end

@inline function (::BoundaryConditionSlipWallMHD)(u_inner, orientation_or_normal,
                                                   direction, x, t,
                                                   surface_flux_functions::Tuple,
                                                   equations::IdealGlmMhdEquations2D)
    flux_fn, noncons_fn = surface_flux_functions
    u_ghost = _mhd_slip_wall_ghost(u_inner, orientation_or_normal)
    if iseven(direction)
        flux_val     = flux_fn(u_inner, u_ghost, orientation_or_normal, equations)
        noncons_flux = noncons_fn(u_inner, u_ghost, orientation_or_normal, equations)
    else
        flux_val     = flux_fn(u_ghost, u_inner, orientation_or_normal, equations)
        noncons_flux = noncons_fn(u_inner, u_ghost, orientation_or_normal, equations)
    end
    return flux_val, noncons_flux
end

const boundary_condition_slip_wall_mhd = BoundaryConditionSlipWallMHD()


###############################################################################
# Stochastic small-scale turbulent body-force driver (Euler domain only).
#
# N Fourier modes are drawn randomly from wavenumber band [kmin, kmax].
# Each mode derives from a stream function
#   ψᵢ = (A/√(nᵢ²+mᵢ²)) sin(nᵢ·kx + φᵢ) cos(mᵢ·ky)
# giving a divergence-free body force (∂f₁/∂x + ∂f₂/∂y = 0):
#   f₁ = -A·mᵢk/√(nᵢ²+mᵢ²) sin(nᵢkx + φᵢ) sin(mᵢky)
#   f₂ = -A·nᵢk/√(nᵢ²+mᵢ²) cos(nᵢkx + φᵢ) cos(mᵢky)
# Normalising by 1/√(n²+m²) gives equal force amplitude A·k for every mode.
# A DiscreteCallback re-randomises all phases φᵢ every τ_corr time units,
# producing a finite correlation time (Ornstein–Uhlenbeck-like behaviour).
###############################################################################

mutable struct TurbulentForcing
    A        :: Float64
    kx       :: Vector{Int}
    ky       :: Vector{Int}
    phases   :: Vector{Float64}
    tau_corr :: Float64
    t_next   :: Float64
    rng      :: Random.MersenneTwister
end

function TurbulentForcing(; A = 1e-2, N = 16, kmin = 4, kmax = 10,
                            tau_corr = 0.1, seed = 42)
    rng    = Random.MersenneTwister(seed)
    kx     = rand(rng, kmin:kmax, N)
    ky     = rand(rng, kmin:kmax, N)
    phases = rand(rng, N) .* 2π
    TurbulentForcing(A, kx, ky, phases, tau_corr, tau_corr, rng)
end

@inline function (f::TurbulentForcing)(u, x, t, equations::CompressibleEulerEquations2D)
    T  = eltype(u)
    k  = convert(T, 2π)
    fa = zero(T)
    fb = zero(T)
    for i in eachindex(f.kx)
        n   = f.kx[i]
        m   = f.ky[i]
        φ   = convert(T, f.phases[i])
        nrm = convert(T, 1 / sqrt(n^2 + m^2))
        fa += -convert(T, f.A) * m * k * nrm * sin(n*k*x[1] + φ) * sin(m*k*x[2])
        fb += -convert(T, f.A) * n * k * nrm * cos(n*k*x[1] + φ) * cos(m*k*x[2])
    end
    rho = max(u[1], eps(T))
    v1  = u[2] / rho
    v2  = u[3] / rho
    return SVector(zero(T), fa, fb, v1*fa + v2*fb)
end

turbulent_forcing = TurbulentForcing(A = 1e-2, N = 16, kmin = 4, kmax = 10,
                                     tau_corr = 0.1, seed = 42)

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

# Bottom half: metallic hydrogen (MHD, semi 1)
mesh_bottom = StructuredMeshView(parent_mesh; indices_min = (1,  1), indices_max = (100,  50))
# Top half:    molecular hydrogen (Euler, semi 2)
mesh_top    = StructuredMeshView(parent_mesh; indices_min = (1, 51), indices_max = (100, 100))

###############################################################################
# Equations and solvers
###############################################################################

equations_mhd   = IdealGlmMhdEquations2D(5 / 3, initial_c_h = 1.0)   # metallic H: γ=5/3
equations_euler = CompressibleEulerEquations2D(7 / 5)                  # molecular H: γ=7/5=1.4

solver_mhd = DGSEM(polydeg = 3,
                   surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                   volume_integral = VolumeIntegralFluxDifferencing(
                       (flux_hindenlang_gassner, flux_nonconservative_powell)))

# Shock-capturing for the Euler (molecular) domain.
# Without explicit viscosity, the turbulent cascade transfers kinetic energy to
# progressively smaller scales until grid-scale Gibbs oscillations drive p or ρ
# negative → NaN.  The Hennemann-Gassner indicator detects cells where the
# solution is under-resolved and blends the DG scheme toward a first-order FV
# scheme there, adding just enough numerical dissipation to prevent NaN without
# over-dissipating smooth regions.  This is more robust than simply increasing
# resolution, because ideal Euler will always eventually cascade to grid scale.
basis_euler     = LobattoLegendreBasis(3)
indicator_euler = IndicatorHennemannGassner(equations_euler, basis_euler;
                                            alpha_max    = 0.5,
                                            alpha_min    = 0.001,
                                            alpha_smooth = false,  # smoothing not implemented for StructuredMeshView
                                            variable     = density_pressure)
solver_euler    = DGSEM(basis_euler, flux_lax_friedrichs,
                        VolumeIntegralShockCapturingHG(indicator_euler;
                                                       volume_flux_dg = flux_ranocha,
                                                       volume_flux_fv = flux_lax_friedrichs))

###############################################################################
# Semidiscretizations
###############################################################################

# Identity coupling functions for periodic x boundaries (self-coupling)
coupling_identity_mhd   = (x, u, equations_other, equations_own) -> u
coupling_identity_euler = (x, u, equations_other, equations_own) -> u

# Bottom (MHD / metallic): periodic in x via self-coupling, slip wall at y_neg,
# coupled to Euler at y_pos.
# x_neg receives from own x_pos (semi index 1, :end in i) and vice versa.
boundary_conditions_bottom = (
    x_neg = BoundaryConditionCoupled(1, (:end,   :i_forward), Float64,
                                     coupling_identity_mhd),
    x_pos = BoundaryConditionCoupled(1, (:begin, :i_forward), Float64,
                                     coupling_identity_mhd),
    y_neg = boundary_condition_slip_wall_mhd,
    y_pos = BoundaryConditionCoupled(2, (:i_forward, :begin), Float64,
                                     coupling_euler_to_mhd),
)

semi_bottom = SemidiscretizationHyperbolic(mesh_bottom, equations_mhd,
                                           initial_condition_mhd, solver_mhd,
                                           boundary_conditions = boundary_conditions_bottom,
                                           source_terms = source_terms_glm_damping)

# Top (Euler / molecular): periodic in x via self-coupling, slip wall at y_pos,
# coupled to MHD at y_neg.
boundary_conditions_top = (
    x_neg = BoundaryConditionCoupled(2, (:end,   :i_forward), Float64,
                                     coupling_identity_euler),
    x_pos = BoundaryConditionCoupled(2, (:begin, :i_forward), Float64,
                                     coupling_identity_euler),
    y_neg = BoundaryConditionCoupled(1, (:i_forward, :end), Float64,
                                     coupling_mhd_to_euler),
    y_pos = boundary_condition_slip_wall,
)

semi_top = SemidiscretizationHyperbolic(mesh_top, equations_euler,
                                        initial_condition_euler, solver_euler,
                                        boundary_conditions = boundary_conditions_top,
                                        source_terms = turbulent_forcing)

semi = SemidiscretizationCoupled(semi_bottom, semi_top)

###############################################################################
# ODE solvers, callbacks
###############################################################################

tspan = (0.0, 72.0)
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

# GLM speed update only for the MHD (metallic, bottom) domain
glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = 0.5, semi_indices = [1])

# Re-randomise all forcing phases every tau_corr time units
forcing_callback = DiscreteCallback(
    (u, t, integrator) -> t >= turbulent_forcing.t_next,
    integrator -> begin
        n = length(turbulent_forcing.phases)
        turbulent_forcing.phases .= rand(turbulent_forcing.rng, n) .* 2π
        turbulent_forcing.t_next  = integrator.t + turbulent_forcing.tau_corr
    end;
    save_positions = (false, false),
)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback,
                        forcing_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0,
            ode_default_options()..., callback = callbacks)

summary_callback()
