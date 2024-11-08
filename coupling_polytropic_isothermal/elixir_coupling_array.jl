using OrdinaryDiffEq
using Trixi

nx = 3
ny = 3

###############################################################################
# semidiscretization of the compressible Euler multicomponent equations
equations = Array{PolytropicEulerEquations2D}(undef, (nx, ny))
for sx in 1:nx
    for sy in 1:ny
        gamma = 2.0
        kappa = 1.0
        if sx == 2 && sy == 2
            gamma = 1.0
#            gamma = 2.0
        end
        equations[sx, sy] = PolytropicEulerEquations2D(gamma, kappa)
    end
end


function initial_condition_wave(x, t, equations::PolytropicEulerEquations2D)
  gamma = equations.gamma
  kappa = equations.kappa

  rho = 1.0
  v1 = 0.0
  if x[1] < -0.5
    rho = ((1.0 + 0.01*sin(x[1]*4*pi)) / kappa)^(1/gamma)
    v1 = ((0.01*sin((x[1]-1/2)*4*pi)) / kappa)
  end
  v2 = 0.0

  return prim2cons(SVector(rho, v1, v2), equations)
end

# general setup
volume_flux = flux_winters_etal
solver = DGSEM(polydeg=3, surface_flux=flux_hll,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

cells_per_dimension = (48, 48)
coordinates_min = (-1.5, -1.5)
coordinates_max = ( 1.5,  1.5)
parent_mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

coupling_function_oo = (x, u, equations_other, equations_own) -> u
coupling_function_oi = (x, u, equations_other, equations_own) -> u
coupling_function_io = (x, u, equations_other, equations_own) -> u
coupling_function_ii = (x, u, equations_other, equations_own) -> u

semis = Array{SemidiscretizationHyperbolic}(undef, (nx, ny))
meshes = Array{StructuredMeshView}(undef, (nx, ny))
for sx in 1:nx
    for sy in 1:ny
        my_idx = sx + (sy - 1)*nx
        left_idx = sx - 1
        right_idx = sx + 1
        top_idx = sy + 1
        bottom_idx = sy - 1

        if (sx - 1 < 1)
            left_idx = sy*nx
        else
            left_idx = my_idx - 1
        end

        if (sx + 1 > nx)
            right_idx = (sy - 1)*nx + 1
        else
            right_idx = my_idx + 1
        end

        if (sy - 1 < 1)
            bottom_idx = sx + nx*(ny - 1)
        else
            bottom_idx = my_idx - ny
        end

        if (sy + 1 > ny)
            top_idx = sx
        else
            top_idx = my_idx + ny
        end

        coordinates_min = (-1.5 + sx-1, -1.5 + sy-1)
        coordinates_max = (-1.5 + sx, -1.5 + sy)
        meshes[sx, sy] = StructuredMeshView(parent_mesh;
                                            indices_min = (16*(sx-1) + 1, 16*(sy - 1) + 1),
                                            indices_max = (16*sx, 16*sy))

        coupling_function1 = coupling_function_oo
        coupling_function2 = coupling_function_oo
        coupling_function3 = coupling_function_oo
        coupling_function4 = coupling_function_oo

        if (left_idx == 2) coupling_function1 = coupling_function_oi end
        if (right_idx == 2) coupling_function2 = coupling_function_oi end
        if (bottom_idx == 2) coupling_function3 = coupling_function_oi end
        if (top_idx == 2) coupling_function4 = coupling_function_oi end

        boundary_conditions = (
                               x_neg=BoundaryConditionCoupled(left_idx, (:end, :i_forward), Float64, coupling_function1),
                               x_pos=BoundaryConditionCoupled(right_idx, (:begin, :i_forward), Float64, coupling_function2),
                               y_neg=BoundaryConditionCoupled(bottom_idx, (:i_forward, :end), Float64, coupling_function3),
                               y_pos=BoundaryConditionCoupled(top_idx, (:i_forward, :begin), Float64, coupling_function4),
#                                x_neg=boundary_condition_periodic,
#                                x_pos=boundary_condition_periodic,
#                                y_neg=boundary_condition_periodic,
#                                y_pos=boundary_condition_periodic,
                              )
        semis[sx, sy] = SemidiscretizationHyperbolic(meshes[sx, sy], equations[sx, sy], initial_condition_wave, solver,
                                                     boundary_conditions=boundary_conditions)
    end
end


# coupled semi
semi = SemidiscretizationCoupled(semis...)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 6.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

# analysis_callback = Array{AnalysisCallback}(undef, (nx, ny))
# for sx in 1:nx
#     for sy in 1:ny
#         analysis_callback[sx, sy] = AnalysisCallback(semis[sx, sy], interval=100)
#     end
# end
# analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback...)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
#                         analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
