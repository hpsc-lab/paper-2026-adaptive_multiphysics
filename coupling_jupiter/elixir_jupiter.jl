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
#   Metallic H (MHD):    ρ=1, p=1, B₂=1 (radial/vertical field from dynamo)
#                        c_s = √(5/3) ≈ 1.29,  v_A = 1.0,  c_fast ≈ 1.63
#   Molecular H (Euler): ρ=1, p=1  (same density/pressure → zero equilibrium flux)
#                        c_s = √(5/3) ≈ 1.29
#
# Equal densities and pressures across the interface are chosen so that the
# coupling functions reproduce the background state exactly and generate no
# spurious interface flux in the absence of waves.  A density jump (ρ_met > ρ_mol)
# would be more realistic but breaks this exact equilibrium; the present setup
# isolates the wave-mode conversion without numerical artifacts.
###############################################################################

function initial_condition_mhd(x, t, equations::IdealGlmMhdEquations2D)
    rho = 1.0
    v1  = 0.0; v2 = 0.0; v3 = 0.0
    p   = 1.0
    B1  = 0.0; B2 = 1.0; B3 = 0.0  # radial guide field (dynamo-generated)
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
# Euler conservative: [ρ, ρv₁, ρv₂, E_euler]
#   E_euler = ρ(v₁²+v₂²)/2 + p/(γ-1)
# MHD  conservative: [ρ, ρv₁, ρv₂, ρv₃, E_mhd, B₁, B₂, B₃, ψ]
#   E_mhd = ρ|v|²/2 + p/(γ-1) + |B|²/2
#
# Equilibrium check (v=0, ρ=p=1, B₂=1):
#   Euler→MHD ghost:  E = 1.5 + 0.5 = 2.0  ✓ matches MHD background
#   MHD→Euler ghost:  E = 2.0 - 0.5 = 1.5  ✓ matches Euler background
###############################################################################

# Euler state → MHD ghost: complete with background radial field B₂=1
coupling_euler_to_mhd = (x, u, equations_other, equations_own) -> begin
    T  = eltype(u)
    B2 = one(T)  # dynamo background field
    SVector(u[1], u[2], u[3], zero(T), u[4] + B2^2 / 2, zero(T), B2, zero(T), zero(T))
end

# MHD state → Euler ghost: strip magnetic energy and out-of-plane kinetic energy
coupling_mhd_to_euler = (x, u, equations_other, equations_own) -> begin
    B_sq_half = (u[6]^2 + u[7]^2 + u[8]^2) / 2
    v3_KE     = u[4]^2 / (2 * max(u[1], eps(eltype(u))))
    SVector(u[1], u[2], u[3], u[5] - B_sq_half - v3_KE)
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
# Absorbing ψ boundary condition (MHD physical boundaries only)
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

equations_mhd   = IdealGlmMhdEquations2D(5 / 3, initial_c_h = 1.0)
equations_euler = CompressibleEulerEquations2D(5 / 3)

solver_mhd = DGSEM(polydeg = 3,
                   surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                   volume_integral = VolumeIntegralFluxDifferencing(
                       (flux_hindenlang_gassner, flux_nonconservative_powell)))

solver_euler = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
                     volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

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
