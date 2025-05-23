using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

"""
Multi-ion system of a reconnection region.
"""


"""
Define the initial condition for the MHD domain as a hyperbolic magnetic field lines
that are being forced to reconnect at the center.
"""
function initial_condition_reconnection(x, t, equations::IdealGlmMhdMultiIonEquations2D)
    rho1 = 1.0
    rho2 = 1.0
    v11 = x[1]/2 * 0.1
    v21 = x[1]/2 * 0.05
    v12 = -x[2]/2 * 0.1
    v22 = -x[2]/2 * 0.05
    v13 = 0.0
    v23 = 0.0
#     p1 = 0.00040170535986
#     p2 = 0.00401705359856
    p1 = 2.0
    p2 = 1.0
    B1 = (-x[1]/2 + x[2])
    B2 = (x[1] + x[2]/2)
    B3 = 0.0
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

# semidiscretization of the ideal MHD equations
equations = IdealGlmMhdMultiIonEquations2D(gammas = (5 / 3, 5 / 3),
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

tspan = (0.0, 20.0) # 100 [ps]

# Entropy conservative volume numerical fluxes with standard LLF dissipation at interfaces
volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)

solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-3.0, -3.0)
coordinates_max = (3.0, 3.0)
# We use a very coarse mesh because this is a 0-dimensional case
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 1_000_000,
                periodicity=(false, false))

# Ion-ion and ion-electron collision source terms
# In this particular case, we can omit source_terms_lorentz because the magnetic field is zero!
function source_terms(u, x, t, equations::IdealGlmMhdMultiIonEquations2D)
    source_terms_collision_ion_ion(u, x, t, equations) +
    source_terms_collision_ion_electron(u, x, t, equations)
end

boundary_condition_neumann_zero = BoundaryConditionNeumann((x, t, equations) -> SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_reconnection, solver,
                                    source_terms = source_terms_lorentz,
#                                     boundary_conditions=BoundaryConditionDirichlet(initial_condition_reconnection))
                                    boundary_conditions=boundary_condition_neumann_zero)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# Define the CFL condition.
cfl = 0.01

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi,
                                     save_analysis = true,
                                     interval = analysis_interval,
                                     extra_analysis_integrals = (temperature1,
                                                                 temperature2))
alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = cfl) # Very small CFL due to the stiff source terms

# The Generalized Lagrange Method divergence cleans the magnetic field.
glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

save_restart = SaveRestartCallback(interval = 1000,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
