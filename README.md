# Plane autopilot

## Characteristics of the aircraft

The considered aircraft is a fighter aircraft of MIRAGE III class.

- Reference length `l_ref = 5.24` m
- Total length `l_t = 3/2 * l_ref` m
- Mass `m = 8400` kg
- position of COG `c = 0.52 \* l_t`
- Surface area `S = 34` m^2
- Radius of gyration `r_g = 2.65` m
- US Standard Atmosphere 76 model
- Altitude `H = 12800` ft
- Mach number `M = 1.04`
- Drag Coefficient for null incidence `C_x0 = 0.03`
- Lift gradient coefficient WRT alpha `C_z_alpha = 3.25`
- Lift gradient coefficient WRT delta `C_z_delta = 1.2`
- Equilibrium fin deflection for null lift `delta_m0 = 0.01`
- Incidence for null lift and null fin deflection `alpha_0 = 0.05`
- Aerodynamic center of body and wings `f = 0.6`
- Aerodynamic center of fins `f_delta = 0.89`
- Polar coefficient `k = 0.3`
- Damping coefficient `Cm_q = -0.35`

## Hypothesis

- Symmetrical flight, in the vertical plane (null sideslip and roll)
- Thrust axis merged with aircraft longitudinal axis
- Inertia principal axis = aircraft transverse axis (diagonal inertiamatrix)
- Fin control loop: its dynamics will be neglected for the controllersynthesis
- The altitude sensor is modeled by a 1st order transfer function with a time constant τ=0.71 s

