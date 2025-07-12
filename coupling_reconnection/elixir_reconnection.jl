using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

"""
Adaptive coupling between a multi-ion MHD system and 8 MHD systems.
"""


"""
Define the initial condition for the MHD domain as a hyperbolic magnetic field lines
that are being forced to reconnect at the center.
"""
function initial_condition_mion(x, t, equations::IdealGlmMhdMultiIonEquations2D)
    rho1 = 1.0
    rho2 = 1.0
    v11 = x[1]/2 * 0.1
    v21 = x[1]/2 * 0.1
    v12 = -x[2]/2 * 0.05
    v22 = -x[2]/2 * 0.05
    v13 = 0.0
    v23 = 0.0
#     p1 = 0.00040170535986
#     p2 = 0.00401705359856
    p1 = 2.0
    p2 = 2.0
    B1 = (-x[1]/2 + x[2])
    B2 = (x[1] + x[2]/2)
    B3 = 0.0
    psi = 0.0

    return prim2cons(SVector(B1, B2, B3, rho1, v11, v12, v13, p1, rho2, v21, v22, v23, p2, psi),
                     equations)
end

"""
Define the initial condition for the MHD domain as a hyperbolic magnetic field lines
that are being forced to reconnect at the center.
"""
function initial_condition_mhd(x, t, equations::IdealGlmMhdEquations2D)
    rho = 2.0
    v1 = x[1]/2 * 0.1
    v2 = -x[2]/2 * 0.05
    v3 = 0.0
    p = 4.0
    B1 = (-x[1]/2 + x[2])
    B2 = (x[1] + x[2]/2)
    B3 = 0.0
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
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

# semidiscretization of the ideal MHD equations
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

equations_mhd = IdealGlmMhdEquations2D(5/3)

# Set up the parent domain.
cells_per_dimension_parent = (96, 96)
coordinates_min = (-3.0, -3.0)
coordinates_max = (3.0, 3.0)
parent_mesh = StructuredMesh(cells_per_dimension_parent, coordinates_min, coordinates_max, periodicity=(false, false))

# Setup up the mesh views.
mesh = Array{StructuredMeshView}(undef, 9)
for mesh_idx_x in 1:3
    for mesh_idx_y in 1:3
        mesh[mesh_idx_x + (mesh_idx_y-1)*3] = StructuredMeshView(parent_mesh;
                                                                 indices_min = (32*(mesh_idx_x-1)+1, 32*(mesh_idx_y-1)+1),
                                                                 indices_max = (32*mesh_idx_x, 32*mesh_idx_y))
    end
end

#     return prim2cons(SVector(B1, B2, B3, rho1, v11, v12, v13, p1, rho2, v21, v22, v23, p2, psi),
#     return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
# Define the coupling functions.
coupling_function_mion_mhd = (x, u, equations_other, equations_own) -> SVector(u[4] + u[9],
                                                                               (u[4]*u[5] + u[9]*u[10])/(u[4] + u[9]),
                                                                               (u[4]*u[6] + u[9]*u[11])/(u[4] + u[9]),
                                                                               (u[4]*u[7] + u[9]*u[12])/(u[4] + u[9]),
                                                                               u[8] + u[13],
                                                                               u[1], u[2], u[3],
                                                                               u[14])
coupling_function_mhd_mion = (x, u, equations_other, equations_own) -> SVector(u[6], u[7], u[8],
                                                                               u[1]/2, u[2], u[3], u[4], u[5]/2,
                                                                               u[1]/2, u[2], u[3], u[4], u[5]/2,
                                                                               u[9])
coupling_function_identity = (x, u, equations_other, equations_own) -> u

# Entropy conservative volume numerical fluxes with standard LLF dissipation at interfaces
volume_flux_moin = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux_mion = (flux_lax_friedrichs, flux_nonconservative_central)
volume_flux_mhd = (flux_hindenlang_gassner, flux_nonconservative_powell)
surface_flux_mhd = (flux_lax_friedrichs, flux_nonconservative_powell)

# Define the semidisretizations.
solver1 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions1 = (x_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                        x_pos=BoundaryConditionCoupled(2, (:begin, :i_forward), Float64, coupling_function_identity),
                        y_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                        y_pos=BoundaryConditionCoupled(4, (:i_forward, :begin), Float64, coupling_function_identity),)
semi1 = SemidiscretizationHyperbolic(mesh[1], equations_mhd, initial_condition_mhd, solver1, boundary_conditions=boundary_conditions1)

solver2 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions2 = (x_neg=BoundaryConditionCoupled(1, (:end, :i_forward), Float64, coupling_function_identity),
                        x_pos=BoundaryConditionCoupled(3, (:begin, :i_forward), Float64, coupling_function_identity),
                        y_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                        y_pos=BoundaryConditionCoupled(5, (:i_forward, :begin), Float64, coupling_function_mion_mhd),)
#                         y_pos=BoundaryConditionCoupled(5, (:i_forward, :begin), Float64, coupling_function_identity),)
semi2 = SemidiscretizationHyperbolic(mesh[2], equations_mhd, initial_condition_mhd, solver2, boundary_conditions=boundary_conditions2)

solver3 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions3 = (x_neg=BoundaryConditionCoupled(2, (:end, :i_forward), Float64, coupling_function_identity),
                        x_pos=BoundaryConditionDirichlet(initial_condition_mhd),
                        y_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                        y_pos=BoundaryConditionCoupled(6, (:i_forward, :begin), Float64, coupling_function_identity),)
semi3 = SemidiscretizationHyperbolic(mesh[3], equations_mhd, initial_condition_mhd, solver3, boundary_conditions=boundary_conditions3)

solver4 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions4 = (x_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                        x_pos=BoundaryConditionCoupled(5, (:begin, :i_forward), Float64, coupling_function_mion_mhd),
#                         x_pos=BoundaryConditionCoupled(5, (:begin, :i_forward), Float64, coupling_function_identity),
                        y_neg=BoundaryConditionCoupled(1, (:i_forward, :end), Float64, coupling_function_identity),
                        y_pos=BoundaryConditionCoupled(7, (:i_forward, :begin), Float64, coupling_function_identity),)
semi4 = SemidiscretizationHyperbolic(mesh[4], equations_mhd, initial_condition_mhd, solver4, boundary_conditions=boundary_conditions4)

solver5 = DGSEM(polydeg = 3, surface_flux = surface_flux_mion,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_moin))
boundary_conditions5 = (x_neg=BoundaryConditionCoupled(4, (:end, :i_forward), Float64, coupling_function_mhd_mion),
                        x_pos=BoundaryConditionCoupled(6, (:begin, :i_forward), Float64, coupling_function_mhd_mion),
                        y_neg=BoundaryConditionCoupled(2, (:i_forward, :end), Float64, coupling_function_mhd_mion),
                        y_pos=BoundaryConditionCoupled(8, (:i_forward, :begin), Float64, coupling_function_mhd_mion),)
semi5 = SemidiscretizationHyperbolic(mesh[5], equations_mion, initial_condition_mion, solver5, boundary_conditions=boundary_conditions5)

# solver5 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
#                 volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
# boundary_conditions5 = (x_neg=BoundaryConditionCoupled(4, (:end, :i_forward), Float64, coupling_function_identity),
#                         x_pos=BoundaryConditionCoupled(6, (:begin, :i_forward), Float64, coupling_function_identity),
#                         y_neg=BoundaryConditionCoupled(2, (:i_forward, :end), Float64, coupling_function_identity),
#                         y_pos=BoundaryConditionCoupled(8, (:i_forward, :begin), Float64, coupling_function_identity),)
# semi5 = SemidiscretizationHyperbolic(mesh[5], equations_mhd, initial_condition_mhd, solver5, boundary_conditions=boundary_conditions5)

solver6 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions6 = (x_neg=BoundaryConditionCoupled(5, (:end, :i_forward), Float64, coupling_function_mion_mhd),
# boundary_conditions6 = (x_neg=BoundaryConditionCoupled(5, (:end, :i_forward), Float64, coupling_function_identity),
                        x_pos=BoundaryConditionDirichlet(initial_condition_mhd),
                        y_neg=BoundaryConditionCoupled(3, (:i_forward, :end), Float64, coupling_function_identity),
                        y_pos=BoundaryConditionCoupled(9, (:i_forward, :begin), Float64, coupling_function_identity),)
semi6 = SemidiscretizationHyperbolic(mesh[6], equations_mhd, initial_condition_mhd, solver6, boundary_conditions=boundary_conditions6)

solver7 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions7 = (x_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                        x_pos=BoundaryConditionCoupled(8, (:begin, :i_forward), Float64, coupling_function_identity),
                        y_neg=BoundaryConditionCoupled(4, (:i_forward, :end), Float64, coupling_function_identity),
                        y_pos=BoundaryConditionDirichlet(initial_condition_mhd),)
semi7 = SemidiscretizationHyperbolic(mesh[7], equations_mhd, initial_condition_mhd, solver7, boundary_conditions=boundary_conditions7)

solver8 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions8 = (x_neg=BoundaryConditionCoupled(7, (:end, :i_forward), Float64, coupling_function_identity),
                        x_pos=BoundaryConditionCoupled(9, (:begin, :i_forward), Float64, coupling_function_identity),
                        y_neg=BoundaryConditionCoupled(5, (:i_forward, :end), Float64, coupling_function_mion_mhd),
#                         y_neg=BoundaryConditionCoupled(5, (:i_forward, :end), Float64, coupling_function_identity),
                        y_pos=BoundaryConditionDirichlet(initial_condition_mhd),)
semi8 = SemidiscretizationHyperbolic(mesh[8], equations_mhd, initial_condition_mhd, solver8, boundary_conditions=boundary_conditions8)

solver9 = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions9 = (x_neg=BoundaryConditionCoupled(8, (:end, :i_forward), Float64, coupling_function_identity),
                        x_pos=BoundaryConditionDirichlet(initial_condition_mhd),
                        y_neg=BoundaryConditionCoupled(6, (:i_forward, :end), Float64, coupling_function_identity),
                        y_pos=BoundaryConditionDirichlet(initial_condition_mhd),)
semi9 = SemidiscretizationHyperbolic(mesh[9], equations_mhd, initial_condition_mhd, solver9, boundary_conditions=boundary_conditions9)

# coupled semidiscretization.
semi = SemidiscretizationCoupled(semi1, semi2, semi3, semi4, semi5, semi6, semi7, semi8, semi9)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0) # 100 [ps]

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# Define the CFL condition.
cfl = 0.01

analysis_interval = 10000
# analysis_callback = AnalysisCallback(semi,
#                                      save_analysis = true,
#                                      interval = analysis_interval,
#                                      extra_analysis_integrals = (temperature1,
#                                                                  temperature2))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = cfl) # Very small CFL due to the stiff source terms

# The Generalized Lagrange Method divergence cleans the magnetic field.
glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl, semi_indices=[1, 2, 3, 4, 5, 6, 7, 8, 9])

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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
