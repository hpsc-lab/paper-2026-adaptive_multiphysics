# Adaptive Multiphysics Coupling Reproduciability Repository

Reproducibility repository for the publication on
"Adaptive Multiphysics Coupling for Hyperbolic Systems".
It contains the Trixi.jl script files (elixirs) used to produce the simulations
shown in the publication, together with the plotting routines written in Python.


## Test Cases

*coupling_euler_polytropic*:
Contains the simulation of two coupled systems, one using the Euler equations
and the other the polytropic equations.

*coupling_polytropic_isothermal*:
Contains the coupled polytropic-isotropic system.
This is a 3x3 array of systems.

*coupling_mhd_ams*:
Contains the coupled MHD-Euler system consisting of eight Euler systems
surrounding one MHD system.
We use adaptive model selection for this one, as we place a magnetic ring
in the center of the MHD domain and advect it in time so that it moves
towards the Euler systems.


## Usage

Our numerical test wer run on Julia 1.10.6.
Older versions, like 1.9, do work, but can give somewhat different performance.
To run any of the three test simulation simply run
```julia
julia> include("coupling_mhd_ams/elixir_euler_mhd_adaptive.jl")
```


## Authors
[Simon Candelaresi](simoncandelaresi.com) and [Michael Schlottke-Lakemper](https://www.uni-augsburg.de/fakultaet/mntf/math/prof/hpsc)
(University of Augsburg, Germany).
