import numpy as np


# ------------ Characteristics of the aircraft --------------

l_ref = 5.24  # Reference length (m)
l_t = 3/2 * l_ref  # Total length (m)
m = 8400  # Mass (kg)
c = 0.52 * l_t  # position of COG (m)
S = 34  # Surface area (m^2)
r_g = 2.65  # Radius of gyration  (m)
H = 12800  # Altitude (ft)
M = 1.04  # Mach number 
C_x0 = 0.03  # Drag Coefficient for null incidence 
C_z_alpha = 3.25  # Lift gradient coefficient WRT alpha 
C_z_delta = 1.2  # Lift gradient coefficient WRT delta 
delta_m0 = 0.01  # Equilibrium fin deflection for null lift 
alpha_0 = 0.05  # Incidence for null lift and null fin deflection 
f = 0.6  # Aerodynamic center of body and wings 
f_delta = 0.89  # Aerodynamic center of fins 
k = 0.3  # Polar coefficient 
Cm_q = -0.35  # Damping coefficient 
V_sound = 324.975 # m/s
rho = 0.827683  # kg/m^3
g0 = 9.81
X_F = f * l_t
X_G = c * l_t
F_delta = f_delta * l_t
X = X_F - X_G
Y = F_delta - X_G


# -------- Algorithm for computing the equilibrium points --------
# Initialization
V_eq = M * V_sound
Q = 0.5 * rho * V_eq **2
alpha_eq = 0
F_p_xeq = 0
epsilon = 0.1 # desired precision in degrees
epsilon = np.radians(epsilon)

while (True):
  alpha_eq_old = alpha_eq

  C_z_eq = 1/(Q*S) * (m*g0 - F_p_xeq * np.sin(alpha_eq))
  C_x_eq = C_x0 + k * C_z_eq ** 2
  F_p_xeq = (Q*S*C_x_eq) / (np.cos(alpha_eq))

  
  C_x_delta = 2 * k * C_z_eq * C_z_delta
 

  delta_m_eq = delta_m0 - (C_x_eq * np.sin(alpha_eq) + C_z_eq * np.cos(alpha_eq)) / (C_x_delta * np.sin(alpha_eq) + C_z_delta * np.cos(alpha_eq)) * (X / (Y-X))

  alpha_eq = alpha_0 + C_z_eq / C_z_alpha - (C_z_delta / C_z_alpha) * delta_m_eq
  print("step")
  if np.abs(alpha_eq - alpha_eq_old) < epsilon:
    break

print(f'alpha eq = {round(alpha_eq, 3)} radians \nF_p_eq = {round(F_p_xeq, 3)} Newtons')