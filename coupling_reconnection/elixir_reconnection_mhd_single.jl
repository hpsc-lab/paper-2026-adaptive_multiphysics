using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

"""
Adaptive coupling between an 9 MHD systems.
"""


###############################################################################
# Semidiscretization of the compressible Euler multicomponent equations.
equations = IdealGlmMhdEquations2D(5/3)

"""
Define the initial condition for the MHD domain as a hyperbolic magnetic field lines
that are being forced to reconnect at the center.
"""
function initial_condition(x, t, equations::IdealGlmMhdEquations2D)
    rho = 1.0
    v1 = x[1]/2
    v2 = -x[2]/2
    v3 = 0.0
    p = rho^equations.gamma
    B1 = -x[1]/2 + x[2]
    B2 = x[1] + x[2]/2
    B3 = 0.0
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

# Set up the parent mesh.
cells_per_dimension_parent = (32, 32)
coordinates_min = (-3.0, -3.0)
coordinates_max = ( 3.0,  3.0)
mesh = StructuredMesh(cells_per_dimension_parent, coordinates_min, coordinates_max, periodicity=(false, false))

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

boundary_conditions = (x_neg=BoundaryConditionDirichlet(initial_condition),
                       x_pos=BoundaryConditionDirichlet(initial_condition),
                       y_neg=BoundaryConditionDirichlet(initial_condition),
                       y_pos=BoundaryConditionDirichlet(initial_condition))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=BoundaryConditionDirichlet(initial_condition))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 8.0.
tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers.
summary_callback = SummaryCallback()

# The SaveSolutionCallback allows to save the solution to a file in regular intervals.
save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory="out",
                                     solution_variables=cons2prim)

# Define the CFL condition.
cfl = 0.1

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=cfl)

# The Generalized Lagrange Method divergence cleans the magnetic field.
glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

# Show that the simulation is still running.
alive_callback = AliveCallback(alive_interval=100)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver.
callbacks = CallbackSet(summary_callback,
                        save_solution,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback,
                        )


###############################################################################
# Run the simulation.

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callback.
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary.
summary_callback()
