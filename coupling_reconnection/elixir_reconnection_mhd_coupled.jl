using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

"""
Adaptive coupling between an 9 MHD systems.
"""


###############################################################################
# Semidiscretization of the compressible Euler multicomponent equations.
equations = IdealGlmMhdEquations2D(5/3)

using StaticArrays

"""
    initial_condition(x, t, equations::IdealGlmMhdEquations2D)

2D Harris current sheet with a small tearing-mode perturbation
A_y(x,z) = -B0*L*log(cosh(x/L)) + δA*cos(k*z)*sech(x/L)

Returned fields are conservative variables via `prim2cons`.
"""
function initial_condition(x, t, equations::IdealGlmMhdEquations2D)
    # --- Parameters ----------------------------------------------------------
    B0   = 1.0          # Asymptotic reversing field
    L    = 0.5          # Sheet half-thickness
    Bg   = 0.1          # Guide field (By)
    n0   = 1.0          # Peak Harris density
    nbg  = 0.2          # Background density
    pbg  = 0.2          # Background pressure

    δA   = 0.01         # Perturbation amplitude in A_y
    k    = 2π / 1.0     # Mode number along z

    γ     = equations.gamma

    # Coordinates
    x_ = x[1]
    z_ = x[2]

    # --- Harris sheet ingredients --------------------------------------------
    sechx = 1 / cosh(x_/L)

    # Vector potential A_y(x,z)
    A_y = -B0 * L * log(cosh(x_/L)) +
          δA * cos(k*z_) * sechx

    # Magnetic field from B = ∇ × (A_y * e_y)
    # Bx = -∂A_y/∂z
    B1 = +δA * k * sin(k*z_) * sechx

    # ∂A_y/∂x
    dAydx = -B0 * tanh(x_/L) +
            δA * cos(k*z_) * (-sechx * tanh(x_/L) / L)

    B3 = dAydx
    B2 = Bg

    # --- Density and pressure (Harris equilibrium) ----------------------------
    rho = n0 * sechx^2 + nbg
    p   = pbg + (B0^2)*sechx^2 / 2   # total pressure balance p+B^2/2 = const

    # --- Velocities (equilibrium, no flow) -----------------------------------
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0

    # GLM divergence-cleaning variable
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end


# Set up the parent mesh.
cells_per_dimension_parent = (96, 96)
coordinates_min = (-3.0, -3.0)
coordinates_max = ( 3.0,  3.0)
parent_mesh = StructuredMesh(cells_per_dimension_parent, coordinates_min, coordinates_max)

# Setup up the mesh views.
mesh = Array{StructuredMeshView}(undef, 9)
for mesh_idx_x in 1:3
    for mesh_idx_y in 1:3
        mesh[mesh_idx_x + (mesh_idx_y-1)*3] = StructuredMeshView(parent_mesh; indices_min = (32*(mesh_idx_x-1)+1, 32*(mesh_idx_y-1)+1),
                                                                 indices_max = (32*mesh_idx_x, 32*mesh_idx_y))
    end
end

# Define the coupling functions.
coupling_function = (x, u, equations_other, equations_own) -> u

# Define the semidisretizations.
solvers = Array{DGSEM}(undef, 9)
semis = Array{SemidiscretizationHyperbolic}(undef, 9)

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)

solvers[1] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions1 = (x_neg=BoundaryConditionDirichlet(initial_condition),
                        x_pos=BoundaryConditionCoupled(2, (:begin, :i_forward), Float64, coupling_function),
                        y_neg=BoundaryConditionDirichlet(initial_condition),
                        y_pos=BoundaryConditionCoupled(4, (:i_forward, :begin), Float64, coupling_function))
semis[1] = SemidiscretizationHyperbolic(mesh[1], equations, initial_condition, solvers[1], boundary_conditions=boundary_conditions1)

solvers[2] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions2 = (x_neg=BoundaryConditionCoupled(1, (:end, :i_forward), Float64, coupling_function),
                        x_pos=BoundaryConditionCoupled(3, (:begin, :i_forward), Float64, coupling_function),
                        y_neg=BoundaryConditionDirichlet(initial_condition),
                        y_pos=BoundaryConditionCoupled(5, (:i_forward, :begin), Float64, coupling_function))
semis[2] = SemidiscretizationHyperbolic(mesh[2], equations, initial_condition, solvers[2], boundary_conditions=boundary_conditions2)

solvers[3] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions3 = (x_neg=BoundaryConditionCoupled(2, (:end, :i_forward), Float64, coupling_function),
                        x_pos=BoundaryConditionDirichlet(initial_condition),
                        y_neg=BoundaryConditionDirichlet(initial_condition),
                        y_pos=BoundaryConditionCoupled(6, (:i_forward, :begin), Float64, coupling_function))
semis[3] = SemidiscretizationHyperbolic(mesh[3], equations, initial_condition, solvers[3], boundary_conditions=boundary_conditions3)

solvers[4] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions4 = (x_neg=BoundaryConditionDirichlet(initial_condition),
                        x_pos=BoundaryConditionCoupled(5, (:begin, :i_forward), Float64, coupling_function),
                        y_neg=BoundaryConditionCoupled(1, (:i_forward, :end), Float64, coupling_function),
                        y_pos=BoundaryConditionCoupled(7, (:i_forward, :begin), Float64, coupling_function))
semis[4] = SemidiscretizationHyperbolic(mesh[4], equations, initial_condition, solvers[4], boundary_conditions=boundary_conditions4)

solvers[5] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions5 = (x_neg=BoundaryConditionCoupled(4, (:end, :i_forward), Float64, coupling_function),
                        x_pos=BoundaryConditionCoupled(6, (:begin, :i_forward), Float64, coupling_function),
                        y_neg=BoundaryConditionCoupled(2, (:i_forward, :end), Float64, coupling_function),
                        y_pos=BoundaryConditionCoupled(8, (:i_forward, :begin), Float64, coupling_function))
semis[5] = SemidiscretizationHyperbolic(mesh[5], equations, initial_condition, solvers[5],
                                     boundary_conditions=boundary_conditions5)

solvers[6] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions6 = (x_neg=BoundaryConditionCoupled(5, (:end, :i_forward), Float64, coupling_function),
                        x_pos=BoundaryConditionDirichlet(initial_condition),
                        y_neg=BoundaryConditionCoupled(3, (:i_forward, :end), Float64, coupling_function),
                        y_pos=BoundaryConditionCoupled(9, (:i_forward, :begin), Float64, coupling_function))
semis[6] = SemidiscretizationHyperbolic(mesh[6], equations, initial_condition, solvers[6], boundary_conditions=boundary_conditions6)

solvers[7] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions7 = (x_neg=BoundaryConditionDirichlet(initial_condition),
                        x_pos=BoundaryConditionCoupled(8, (:begin, :i_forward), Float64, coupling_function),
                        y_neg=BoundaryConditionCoupled(4, (:i_forward, :end), Float64, coupling_function),
                        y_pos=BoundaryConditionDirichlet(initial_condition))
semis[7] = SemidiscretizationHyperbolic(mesh[7], equations, initial_condition, solvers[7], boundary_conditions=boundary_conditions7)

solvers[8] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions8 = (x_neg=BoundaryConditionCoupled(7, (:end, :i_forward), Float64, coupling_function),
                        x_pos=BoundaryConditionCoupled(9, (:begin, :i_forward), Float64, coupling_function),
                        y_neg=BoundaryConditionCoupled(5, (:i_forward, :end), Float64, coupling_function),
                        y_pos=BoundaryConditionDirichlet(initial_condition))
semis[8] = SemidiscretizationHyperbolic(mesh[8], equations, initial_condition, solvers[8], boundary_conditions=boundary_conditions8)

solvers[9] = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions9 = (x_neg=BoundaryConditionCoupled(8, (:end, :i_forward), Float64, coupling_function),
                        x_pos=BoundaryConditionDirichlet(initial_condition),
                        y_neg=BoundaryConditionCoupled(6, (:i_forward, :end), Float64, coupling_function),
                        y_pos=BoundaryConditionDirichlet(initial_condition))
semis[9] = SemidiscretizationHyperbolic(mesh[9], equations, initial_condition, solvers[9], boundary_conditions=boundary_conditions9)

# Put together the coupled semidiscretization.
semi = SemidiscretizationCoupled(semis...)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 8.0.
tspan = (0.0, 8.0)
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
glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl, semi_indices=[1, 2, 3, 4, 5, 6, 7, 8, 9])

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
