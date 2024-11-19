# paper-2024-structured_mesh_coupled
Reproducibility repository for the publication on
"Adaptive Multiphysics Coupling for Hyperbolic Systems".
It contains the Trixi.jl script files used for produce the simulations
shown in the publication, together with the plotting routines.

*coupling_euler_polytropic*:
Contains the simulation of two coupled systems, one using the Euler equations
and the other the polytropic equations.

*coupling_polytropic_isothermal*:
Contains the coupled polytropic-isotropic system.
This is a 3x3 array of systems.

*coupling_mhd_ams*:
Contains the coupled MHD-Euler system consisting of eight Euler systems
surrounding an MHD system.
We use adaptive model selection for this one, as we place a magnetic ring
in the center of the MHD domain and advect it in time.
