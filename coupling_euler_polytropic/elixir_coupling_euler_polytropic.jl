using OrdinaryDiffEq
using Trixi

"""
Coupled one polytropic Euler system with one Euler system.
"""

###############################################################################
# semidiscretization of the compressible Euler multicomponent equations
gamma1 = 2.0
kappa1 = 1.0
equations1 = PolytropicEulerEquations2D(gamma1, kappa1)
equations2 = CompressibleEulerEquations2D(5/3)

function initial_condition_wave(x, t, equations::PolytropicEulerEquations2D)
    gamma = equations.gamma
    kappa = equations.kappa

    rho = 1.0
    v1 = 0.0
    if x[1] < 0.0
        rho = ((1.0 + 0.01*sin(x[1]*2*pi)) / kappa)^(1/gamma)
        v1 = -((0.01*sin((x[1]-1/2)*2*pi)) / kappa)
    end
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2), equations)
end

function initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    p = rho.^equations.gamma

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

# set up the parent mesh
cells_per_dimension_parent = (64, 32)
coordinates_min = (-2.0, -1.0)
coordinates_max = ( 2.0,  1.0)
parent_mesh = StructuredMesh(cells_per_dimension_parent, coordinates_min, coordinates_max)

# Setup up the mesh views.
mesh1 = StructuredMeshView(parent_mesh; indices_min = (1, 1), indices_max = (32, 32))
mesh2 = StructuredMeshView(parent_mesh; indices_min = (33, 1), indices_max = (64, 32))

# Extend the definition of the non-conservative Powell flux functions.
import Trixi.flux_nonconservative_powell
function flux_nonconservative_powell(u_ll, u_rr,
                                     normal_direction_ll::AbstractVector,
                                     equations::IdealGlmMhdEquations2D)
    flux_nonconservative_powell(u_ll, u_rr, normal_direction_ll, normal_direction_ll,
                                equations)
end

# Define the solver for both systems.
volume_flux = flux_winters_etal

# semi 1
initial_condition1 = initial_condition_wave

volume_flux = flux_winters_etal
solver1 = DGSEM(polydeg=3, surface_flux=flux_hll,
                volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coupling_function1 = (x, u, equations_other, equations_own) -> SVector(u[1], u[2], u[3])
boundary_conditions1 = (
                       x_neg=BoundaryConditionCoupled(2, (:end, :i_forward), Float64, coupling_function1),
                       x_pos=BoundaryConditionCoupled(2, (:begin, :i_forward), Float64, coupling_function1),
                       y_neg=boundary_condition_periodic,
                       y_pos=boundary_condition_periodic,
                      )
semi1 = SemidiscretizationHyperbolic(mesh1, equations1, initial_condition1, solver1,
                                     boundary_conditions=boundary_conditions1)

# semi 2
initial_condition2 = initial_condition_constant

solver2 = DGSEM(polydeg = 3, surface_flux = flux_hll,
                volume_integral = VolumeIntegralWeakForm())

# coupling_function2 = (x, u, equations_other, equations_own) -> SVector(u[1], u[2], u[3], u[1]^equations_own.gamma)
coupling_function2 = (x, u, equations_other, equations_own) -> SVector(u[1], u[2], u[3], u[1]^equations_own.gamma * equations_own.inv_gamma_minus_one + 0.5 * (u[2]^2/u[1] + u[3]^2/u[1]))
boundary_conditions2 = (
                       x_neg=BoundaryConditionCoupled(1, (:end, :i_forward), Float64, coupling_function2),
                       x_pos=BoundaryConditionCoupled(1, (:begin, :i_forward), Float64, coupling_function2),
                       y_neg=boundary_condition_periodic,
                       y_pos=boundary_condition_periodic,
                      )
semi2 = SemidiscretizationHyperbolic(mesh2, equations2, initial_condition2, solver2,
                                     boundary_conditions=boundary_conditions2)

# coupled semi
semi = SemidiscretizationCoupled(semi1, semi2)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback1 = AnalysisCallback(semi1, interval=100)
analysis_callback2 = AnalysisCallback(semi2, interval=100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=5,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

cfl = 0.1

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
#                         analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        )


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

