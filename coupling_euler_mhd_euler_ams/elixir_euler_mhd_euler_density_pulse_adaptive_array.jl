using OrdinaryDiffEq
using Trixi

function ams_criteria!(u_old, semi_old, t_expand, t, shrink_delay)
    # Determine if we should remesh.
    e_mag = u_old[5][6, :, :, :].^2 + u_old[5][7, :, :, :].^2
    e_mag = reshape(e_mag, (4, 4, semi_old[5].mesh.cells_per_dimension[1], semi_old[5].mesh.cells_per_dimension[2]))
    mhd_index_delta_left = 0
    mhd_index_delta_right = 0
    mhd_index_delta_up = 0
    mhd_index_delta_down = 0

    if maximum(e_mag[:, :, begin, :]) < 1e-8
        if (t_expand[1] == 0) || (t - t_expand[1] >= shrink_delay)
            mhd_index_delta_left = 1
            t_expand[1] = t
        end
    end
    if maximum(e_mag[:, :, begin, :]) > 2e-7
        if (t_expand[1] == 0) || (t - t_expand[1] >= shrink_delay)
            mhd_index_delta_left = -1
            t_expand[1] = t
        end
    end
    if maximum(e_mag[:, :, end, :]) < 1e-8
        if (t_expand[2] == 0) || (t - t_expand[2] >= shrink_delay)
            mhd_index_delta_right = -1
            t_expand[2] = t
        end
    end
    if maximum(e_mag[:, :, end, :]) > 2e-7
        if (t_expand[2] == 0) || (t - t_expand[2] >= shrink_delay)
            mhd_index_delta_right = 1
            t_expand[2] = t
        end
    end
    if maximum(e_mag[:, :, :, begin]) < 1e-8
        if (t_expand[3] == 0) || (t - t_expand[3] >= shrink_delay)
            mhd_index_delta_down = 1
            t_expand[3] = t
        end
    end
    if maximum(e_mag[:, :, :, begin]) > 2e-7
        if (t_expand[3] == 0) || (t - t_expand[3] >= shrink_delay)
            mhd_index_delta_down = -1
            t_expand[3] = t
        end
    end
    if maximum(e_mag[:, :, :, end]) < 1e-8
        if (t_expand[4] == 0) || (t - t_expand[4] >= shrink_delay)
            mhd_index_delta_up = -1
            t_expand[4] = t
        end
    end
    if maximum(e_mag[:, :, :, end]) > 2e-7
        if (t_expand[4] == 0) || (t - t_expand[4] >= shrink_delay)
            mhd_index_delta_up = 1
            t_expand[4] = t
        end
    end

    return (mhd_index_delta_left, mhd_index_delta_right, mhd_index_delta_up, mhd_index_delta_down)
end

###############################################################################
# define the callbacks changing the mesh min and max indices
struct AmsCallback
    parent_mesh
    mesh
    shrink_delay
    times::Vector{Float64}
    min_values::Vector{Float64}
    max_values::Vector{Float64}
    t_expand::Vector{Float64}

    # You can optionally define an inner constructor like the one below to set up
    # some required stuff. You can also create outer constructors (not demonstrated
    # here) for further customization options.
    function AmsCallback(parent_mesh, shrink_delay)
        new(parent_mesh, mesh, shrink_delay, Float64[], Float64[], Float64[], Vector{Float64}([0, 0, 0, 0]))
    end
end

function (ams_callback::AmsCallback)(integrator)
    elapsed_time = @elapsed begin
        u_ode = integrator.u
        t = integrator.t

        min_val, max_val = extrema(u_ode)
        push!(ams_callback.times, t)
        push!(ams_callback.min_values, min_val)
        push!(ams_callback.max_values, max_val)

        parent_mesh = ams_callback.parent_mesh
        shrink_delay = ams_callback.shrink_delay
        t_expand = ams_callback.t_expand
        mesh = ams_callback.mesh

        semi_old = Array{SemidiscretizationHyperbolic}(undef, 9)
        for mesh_idx in 1:9
            semi_old[mesh_idx] = integrator.p.semis[mesh_idx]
        end

        node_coordinates_old = Array{typeof(semi_old[1].cache.elements.node_coordinates)}(undef, 9)
        for mesh_idx in 1:9
            node_coordinates_old[mesh_idx] = semi_old[mesh_idx].cache.elements.node_coordinates
        end

        # Get the u-values in a reshaped and more menegable form.
        u_indices_old = integrator.p.u_indices
        u_old = Array{Array}(undef, 9)
        for mesh_idx in 1:9
            if mesh_idx != 5
                u_old[mesh_idx] = reshape(integrator.u[u_indices_old[mesh_idx]], (4, 4, 4, prod(semi_old[mesh_idx].mesh.cells_per_dimension)))
            else
                u_old[mesh_idx] = reshape(integrator.u[u_indices_old[mesh_idx]], (9, 4, 4, prod(semi_old[mesh_idx].mesh.cells_per_dimension)))
            end
        end

#         # Determine if we should remesh.
#         e_mag = u_old[5][6, :, :, :].^2 + u_old[5][7, :, :, :].^2
#         e_mag = reshape(e_mag, (4, 4, semi_old[5].mesh.cells_per_dimension[1], semi_old[5].mesh.cells_per_dimension[2]))
#         mhd_index_delta_left = 0
#         mhd_index_delta_right = 0
#         mhd_index_delta_up = 0
#         mhd_index_delta_down = 0
#
#         if maximum(e_mag[:, :, begin, :]) < 1e-8
#             if (t_expand[1] == 0) || (t - t_expand[1] >= shrink_delay)
#                 mhd_index_delta_left = 1
#                 t_expand[1] = t
#             end
#         end
#         if maximum(e_mag[:, :, begin, :]) > 2e-7
#             if (t_expand[1] == 0) || (t - t_expand[1] >= shrink_delay)
#                 mhd_index_delta_left = -1
#                 t_expand[1] = t
#             end
#         end
#         if maximum(e_mag[:, :, end, :]) < 1e-8
#             if (t_expand[2] == 0) || (t - t_expand[2] >= shrink_delay)
#                 mhd_index_delta_right = -1
#                 t_expand[2] = t
#             end
#         end
#         if maximum(e_mag[:, :, end, :]) > 2e-7
#             if (t_expand[2] == 0) || (t - t_expand[2] >= shrink_delay)
#                 mhd_index_delta_right = 1
#                 t_expand[2] = t
#             end
#         end
#         if maximum(e_mag[:, :, :, begin]) < 1e-8
#             if (t_expand[3] == 0) || (t - t_expand[3] >= shrink_delay)
#                 mhd_index_delta_down = 1
#                 t_expand[3] = t
#             end
#         end
#         if maximum(e_mag[:, :, :, begin]) > 2e-7
#             if (t_expand[3] == 0) || (t - t_expand[3] >= shrink_delay)
#                 mhd_index_delta_down = -1
#                 t_expand[3] = t
#             end
#         end
#         if maximum(e_mag[:, :, :, end]) < 1e-8
#             if (t_expand[4] == 0) || (t - t_expand[4] >= shrink_delay)
#                 mhd_index_delta_up = -1
#                 t_expand[4] = t
#             end
#         end
#         if maximum(e_mag[:, :, :, end]) > 2e-7
#             if (t_expand[4] == 0) || (t - t_expand[4] >= shrink_delay)
#                 mhd_index_delta_up = 1
#                 t_expand[4] = t
#             end
#         end

        (mhd_index_delta_left, mhd_index_delta_right, mhd_index_delta_up, mhd_index_delta_down) = ams_criteria!(u_old, semi_old, t_expand, t, shrink_delay)

        # Perform the remeshing.
        if any((mhd_index_delta_left, mhd_index_delta_right, mhd_index_delta_down, mhd_index_delta_up) .!= 0 )
            println((mhd_index_delta_left, mhd_index_delta_right, mhd_index_delta_down, mhd_index_delta_up))
            elapsed_time_ams = @elapsed begin
            # Generate the new resized meshes.
            mesh_new = Array{StructuredMeshView}(undef, 9)
            mesh_new[1] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (1, 1),
                                             indices_max = (mesh[1].indices_max[1]+mhd_index_delta_left, mesh[1].indices_max[2]+mhd_index_delta_down))
            mesh_new[2] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (mesh[2].indices_min[1]+mhd_index_delta_left, 1),
                                             indices_max = (mesh[2].indices_max[1]+mhd_index_delta_right, mesh[2].indices_max[2]+mhd_index_delta_down))
            mesh_new[3] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (mesh[3].indices_min[1]+mhd_index_delta_right, 1),
                                             indices_max = (96, mesh[3].indices_max[2]+mhd_index_delta_down))
            mesh_new[4] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (1, mesh[4].indices_min[2]+mhd_index_delta_down),
                                             indices_max = (mesh[4].indices_max[1]+mhd_index_delta_left, mesh[4].indices_max[2]+mhd_index_delta_up))
            mesh_new[5] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (mesh[5].indices_min[1]+mhd_index_delta_left, mesh[5].indices_min[2]+mhd_index_delta_down),
                                             indices_max = (mesh[5].indices_max[1]+mhd_index_delta_right, mesh[5].indices_max[2]+mhd_index_delta_up))
            mesh_new[6] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (mesh[6].indices_min[1]+mhd_index_delta_right, mesh[6].indices_min[2]+mhd_index_delta_down),
                                             indices_max = (96, mesh[6].indices_max[2]+mhd_index_delta_up))
            mesh_new[7] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (1, mesh[7].indices_min[2]+mhd_index_delta_up),
                                             indices_max = (mesh[7].indices_max[1]+mhd_index_delta_left, 96))
            mesh_new[8] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (mesh[8].indices_min[1]+mhd_index_delta_left, mesh[8].indices_min[2]+mhd_index_delta_up),
                                             indices_max = (mesh[8].indices_max[1]+mhd_index_delta_right, 96))
            mesh_new[9] = StructuredMeshView(ams_callback.parent_mesh;
                                             indices_min = (mesh[9].indices_min[1]+mhd_index_delta_right, mesh[9].indices_min[2]+mhd_index_delta_up),
                                             indices_max = (96, 96))

            iter_old = integrator.iter
            naccept_old = integrator.stats.naccept

            # Define the semidiscretizations.
            semis = Array{SemidiscretizationHyperbolic}(undef, 9)
            for mesh_idx in 1:9
                semis[mesh_idx] = SemidiscretizationHyperbolic(mesh_new[mesh_idx], semi_old[mesh_idx].equations,
                                                               semi_old[mesh_idx].initial_condition,
                                                               semi_old[mesh_idx].solver,
                                                               boundary_conditions = semi_old[mesh_idx].boundary_conditions)
            end
            semi = SemidiscretizationCoupled(semis[1], semis[2], semis[3], semis[4], semis[5],
                                             semis[6], semis[7], semis[8], semis[9])

            # Define the new ODE.
            ode = semidiscretize(semi, (integrator.t, integrator.opts.tstops.valtree[1]))

            # Get the new u-arrays using the new format.
            u_new = Vector{Float64}(undef, semi.u_indices[9][end])
            u = Array{Array}(undef, 9)
            for mesh_idx in 1:9
                if mesh_idx != 5
                    u[mesh_idx] = reshape(u_new[ode.p.u_indices[mesh_idx]], (4, 4, 4, size(semis[mesh_idx].cache.elements.node_coordinates)[4]))
                else
                    u[mesh_idx] = reshape(u_new[ode.p.u_indices[mesh_idx]], (9, 4, 4, size(semis[mesh_idx].cache.elements.node_coordinates)[4]))
                end
            end

            for semi_idx in 1:9
                for semi_old_idx in 1:9
                    for element_old in 1:size(node_coordinates_old[semi_old_idx])[4]
                        same_coordinates = abs.(semis[semi_idx].cache.elements.node_coordinates[:, 1, 1, :] .- node_coordinates_old[semi_old_idx][:, 1, 1, element_old]) .< 1e-10
                        element = findall(x -> x==true, same_coordinates[1, :] .* same_coordinates[2, :])
                        if length(element) > 0
                            element = element[1]
                            if (semi_idx != 5 && semi_old_idx != 5)
                                # Old and new Euler
                                u[semi_idx][:, :, :, element] .= u_old[semi_old_idx][:, :, :, element_old]
                            elseif (semi_idx == 5 && semi_old_idx == 5)
                                # Old and new MHD
                                u[semi_idx][:, :, :, element] .= u_old[semi_old_idx][:, :, :, element_old]
                            elseif (semi_idx != 5 && semi_old_idx == 5)
                                # Old MHD and new Euler
                                # Copy density and velocity.
                                u[semi_idx][1:3, :, :, element] .= u_old[semi_old_idx][1:3, :, :, element_old]
                                # Copy the pressure/temperature
                                u[semi_idx][4, :, :, element] .= u_old[semi_old_idx][5, :, :, element_old]
                            else
                                # Old Euler and new MHD
                                # Copy density and velocity.
                                u[semi_idx][1:3, :, :, element] .= u_old[semi_old_idx][1:3, :, :, element_old]
                                # Copy the pressure/temperature
                                u[semi_idx][5, :, :, element] .= u_old[semi_old_idx][4, :, :, element_old]
                                #  Set non-matching variables are 0.
                                u[semi_idx][4, :, :, element] .= 0.0
                                u[semi_idx][6:end, :, :, element] .= 0.0
                            end
                        end
                    end
                end
            end

            resize!(integrator.u, size(u_new)[1])
            for mesh_idx in 1:9
                integrator.u[ode.p.u_indices[mesh_idx]] = u[mesh_idx][:]
            end

            integrator.iter = iter_old
            integrator.stats.naccept = naccept_old

            integrator.p = ode.p

            for mesh_idx in 1:9
                mesh[mesh_idx].indices_min = mesh_new[mesh_idx].indices_min
                mesh[mesh_idx].indices_max = mesh_new[mesh_idx].indices_max
            end
        end

        for mesh_idx in 1:9
            integrator.p.semis[mesh_idx].mesh.unsaved_changes = true
            mesh[mesh_idx].unsaved_changes = true
        end
        parent_mesh.unsaved_changes = true

        # avoid re-evaluating possible FSAL stages
        u_modified!(integrator, false)
        println("elapsed_time_ams = ", elapsed_time_ams)
        end
    end

    println("elapsed_time = ", elapsed_time)

    return nothing
end


function AmsCallback(; parent_mesh, mesh, shrink_delay)
    # Call the `AmsCallback` after every RK step.
    condition = (u_ode, t, integrator) -> true

    ams_callback = AmsCallback(parent_mesh, shrink_delay)

    DiscreteCallback(condition, ams_callback,
                     save_positions=(false, false))
end

###############################################################################
# semidiscretization of the compressible Euler multicomponent equations
equations_euler = CompressibleEulerEquations2D(5/3)
equations_mhd = IdealGlmMhdEquations2D(5/3)

function initial_condition_bump(x, t, equations::IdealGlmMhdEquations2D)
    rho = 1.0
    v1 = 0.2
    v2 = 0.1
    v3 = 0.0
    p = rho^equations.gamma
    r = sqrt(x[1]^2 + x[2]^2)
    B1 = x[2] * exp(-r^2*10)
    B2 = -x[1] * exp(-r^2*10)
    B3 = 0.0
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

function initial_condition_quiet(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    v1 = 0.2
    v2 = 0.1
    p = rho.^equations.gamma

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

# set up the parent mesh
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
coupling_function_euler_euler = (x, u, equations_other, equations_own) -> u
coupling_function_mhd_euler = (x, u, equations_other, equations_own) -> SVector(u[1], u[2], u[3], u[5])
coupling_function_euler_mhd = (x, u, equations_other, equations_own) -> SVector(u[1], u[2], u[3], 0.0, u[4], 0.0, 0.0, 0.0, 0.0)

# Define the semidisretizations.
solver1 = DGSEM(polydeg = 3, surface_flux = flux_hll, volume_integral = VolumeIntegralWeakForm())
boundary_conditions1 = (x_neg=BoundaryConditionCoupled(3, (:end, :i_forward), Float64, coupling_function_euler_euler),
                        x_pos=BoundaryConditionCoupled(2, (:begin, :i_forward), Float64, coupling_function_euler_euler),
                        y_neg=BoundaryConditionCoupled(7, (:i_forward, :end), Float64, coupling_function_euler_euler),
                        y_pos=BoundaryConditionCoupled(4, (:i_forward, :begin), Float64, coupling_function_euler_euler),)
semi1 = SemidiscretizationHyperbolic(mesh[1], equations_euler, initial_condition_quiet, solver1, boundary_conditions=boundary_conditions1)

solver2 = DGSEM(polydeg = 3, surface_flux = flux_hll, volume_integral = VolumeIntegralWeakForm())
boundary_conditions2 = (x_neg=BoundaryConditionCoupled(1, (:end, :i_forward), Float64, coupling_function_euler_euler),
                        x_pos=BoundaryConditionCoupled(3, (:begin, :i_forward), Float64, coupling_function_euler_euler),
                        y_neg=BoundaryConditionCoupled(8, (:i_forward, :end), Float64, coupling_function_euler_euler),
                        y_pos=BoundaryConditionCoupled(5, (:i_forward, :begin), Float64, coupling_function_mhd_euler),)
semi2 = SemidiscretizationHyperbolic(mesh[2], equations_euler, initial_condition_quiet, solver2, boundary_conditions=boundary_conditions2)

solver3 = DGSEM(polydeg = 3, surface_flux = flux_hll, volume_integral = VolumeIntegralWeakForm())
boundary_conditions3 = (x_neg=BoundaryConditionCoupled(2, (:end, :i_forward), Float64, coupling_function_euler_euler),
                        x_pos=BoundaryConditionCoupled(1, (:begin, :i_forward), Float64, coupling_function_euler_euler),
                        y_neg=BoundaryConditionCoupled(9, (:i_forward, :end), Float64, coupling_function_euler_euler),
                        y_pos=BoundaryConditionCoupled(6, (:i_forward, :begin), Float64, coupling_function_euler_euler),)
semi3 = SemidiscretizationHyperbolic(mesh[3], equations_euler, initial_condition_quiet, solver3, boundary_conditions=boundary_conditions3)

solver4 = DGSEM(polydeg = 3, surface_flux = flux_hll, volume_integral = VolumeIntegralWeakForm())
boundary_conditions4 = (x_neg=BoundaryConditionCoupled(6, (:end, :i_forward), Float64, coupling_function_euler_euler),
                        x_pos=BoundaryConditionCoupled(5, (:begin, :i_forward), Float64, coupling_function_mhd_euler),
                        y_neg=BoundaryConditionCoupled(1, (:i_forward, :end), Float64, coupling_function_euler_euler),
                        y_pos=BoundaryConditionCoupled(7, (:i_forward, :begin), Float64, coupling_function_euler_euler),)
semi4 = SemidiscretizationHyperbolic(mesh[4], equations_euler, initial_condition_quiet, solver4, boundary_conditions=boundary_conditions4)

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver5 = DGSEM(polydeg = 3, surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions5 = (x_neg=BoundaryConditionCoupled(4, (:end, :i_forward), Float64, coupling_function_euler_mhd),
                        x_pos=BoundaryConditionCoupled(6, (:begin, :i_forward), Float64, coupling_function_euler_mhd),
                        y_neg=BoundaryConditionCoupled(2, (:i_forward, :end), Float64, coupling_function_euler_mhd),
                        y_pos=BoundaryConditionCoupled(8, (:i_forward, :begin), Float64, coupling_function_euler_mhd),)
semi5 = SemidiscretizationHyperbolic(mesh[5], equations_mhd, initial_condition_bump, solver5,
                                     boundary_conditions=boundary_conditions5)

solver6 = DGSEM(polydeg = 3, surface_flux = flux_hll, volume_integral = VolumeIntegralWeakForm())
boundary_conditions6 = (x_neg=BoundaryConditionCoupled(5, (:end, :i_forward), Float64, coupling_function_mhd_euler),
                        x_pos=BoundaryConditionCoupled(4, (:begin, :i_forward), Float64, coupling_function_euler_euler),
                        y_neg=BoundaryConditionCoupled(3, (:i_forward, :end), Float64, coupling_function_euler_euler),
                        y_pos=BoundaryConditionCoupled(9, (:i_forward, :begin), Float64, coupling_function_euler_euler),)
semi6 = SemidiscretizationHyperbolic(mesh[6], equations_euler, initial_condition_quiet, solver6, boundary_conditions=boundary_conditions6)

solver7 = DGSEM(polydeg = 3, surface_flux = flux_hll, volume_integral = VolumeIntegralWeakForm())
boundary_conditions7 = (x_neg=BoundaryConditionCoupled(9, (:end, :i_forward), Float64, coupling_function_euler_euler),
                        x_pos=BoundaryConditionCoupled(8, (:begin, :i_forward), Float64, coupling_function_euler_euler),
                        y_neg=BoundaryConditionCoupled(4, (:i_forward, :end), Float64, coupling_function_euler_euler),
                        y_pos=BoundaryConditionCoupled(1, (:i_forward, :begin), Float64, coupling_function_euler_euler),)
semi7 = SemidiscretizationHyperbolic(mesh[7], equations_euler, initial_condition_quiet, solver7, boundary_conditions=boundary_conditions7)

solver8 = DGSEM(polydeg = 3, surface_flux = flux_hll, volume_integral = VolumeIntegralWeakForm())
boundary_conditions8 = (x_neg=BoundaryConditionCoupled(7, (:end, :i_forward), Float64, coupling_function_euler_euler),
                        x_pos=BoundaryConditionCoupled(9, (:begin, :i_forward), Float64, coupling_function_euler_euler),
                        y_neg=BoundaryConditionCoupled(5, (:i_forward, :end), Float64, coupling_function_mhd_euler),
                        y_pos=BoundaryConditionCoupled(2, (:i_forward, :begin), Float64, coupling_function_euler_euler),)
semi8 = SemidiscretizationHyperbolic(mesh[8], equations_euler, initial_condition_quiet, solver8, boundary_conditions=boundary_conditions8)

solver9 = DGSEM(polydeg = 3, surface_flux = flux_hll, volume_integral = VolumeIntegralWeakForm())
boundary_conditions9 = (x_neg=BoundaryConditionCoupled(8, (:end, :i_forward), Float64, coupling_function_euler_euler),
                        x_pos=BoundaryConditionCoupled(7, (:begin, :i_forward), Float64, coupling_function_euler_euler),
                        y_neg=BoundaryConditionCoupled(6, (:i_forward, :end), Float64, coupling_function_euler_euler),
                        y_pos=BoundaryConditionCoupled(3, (:i_forward, :begin), Float64, coupling_function_euler_euler),)
semi9 = SemidiscretizationHyperbolic(mesh[9], equations_euler, initial_condition_quiet, solver9, boundary_conditions=boundary_conditions9)

# coupled semi
semi = SemidiscretizationCoupled(semi1, semi2, semi3, semi4, semi5, semi6, semi7, semi8, semi9)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 8.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# The min_max indices callback changes the min/max indices of the meshe views dynamically.
ams_callback = AmsCallback(parent_mesh=parent_mesh, mesh=mesh, shrink_delay=0.2)

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

cfl = 0.1

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl, semi_indices=[5])

callbacks = CallbackSet(summary_callback,
#                         analysis_callback,
                        alive_callback,
                        save_solution,
                        ams_callback,
                        stepsize_callback,
                        glm_speed_callback,
                        )


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
