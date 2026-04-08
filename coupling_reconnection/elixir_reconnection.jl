using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

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
    p_thermal = 2*beta*p_mag
    p = p_thermal + p_mag

    rho = p

    # Perturbation of the velocity.
    v1 = 0.0
    v2 = 0.0
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
    p_thermal = 2*beta*p_mag
    p1 = (p_thermal + p_mag)/2
    p2 = (p_thermal + p_mag)/2

    rho1 = p1
    rho2 = p2

    # Perturbation of the velocity.
    v11 = 0.0
    v12 = 0.0
    v13 = 0.0
    v21 = 0.0
    v22 = 0.0
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
volume_flux_mion = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux_mion = (flux_lax_friedrichs, flux_nonconservative_central)
volume_flux_mhd = (flux_hindenlang_gassner, flux_nonconservative_powell)
surface_flux_mhd = (flux_lax_friedrichs, flux_nonconservative_powell)

# Define the semidisretizations.
solver_bottom = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                      volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
boundary_conditions_bottom = (x_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                              x_pos=BoundaryConditionDirichlet(initial_condition_mhd),
                              y_neg=BoundaryConditionDirichlet(initial_condition_mhd),
                              y_pos=BoundaryConditionCoupled(2, (:i_forward, :begin), Float64, coupling_function_mion_mhd),)
semi_bottom = SemidiscretizationHyperbolic(mesh_bottom, equations_mhd,
                                           initial_condition_mhd, solver_bottom,
                                           boundary_conditions=boundary_conditions_bottom)

solver_middle = DGSEM(polydeg = 3, surface_flux = surface_flux_mion,
                      volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mion))
boundary_conditions_middle = (x_neg=BoundaryConditionDirichlet(initial_condition_mionmhd),
                              x_pos=BoundaryConditionDirichlet(initial_condition_mionmhd),
                              y_neg=BoundaryConditionCoupled(1, (:i_forward, :end), Float64, coupling_function_mhd_mion),
                              y_pos=BoundaryConditionCoupled(3, (:i_forward, :begin), Float64, coupling_function_mhd_mion),)
semi_middle = SemidiscretizationHyperbolic(mesh_middle, equations_mion,
                                           initial_condition_mionmhd, solver_middle,
                                           boundary_conditions=boundary_conditions_middle)

solver_top = DGSEM(polydeg = 3, surface_flux = surface_flux_mhd,
                   volume_integral = VolumeIntegralFluxDifferencing(volume_flux_mhd))
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
glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl, semi_indices=[1, 2, 3])

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
