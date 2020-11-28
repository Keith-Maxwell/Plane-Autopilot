import numpy as np
from control import *
from control.matlab import *
import matplotlib.pyplot as plt
from sisopy31 import *

np.set_printoptions(4, suppress=True)

# ------------ Characteristics of the aircraft --------------

l_ref = 5.24  # Reference length (m)
l_t = 3 / 2 * l_ref  # Total length (m)
m = 8400  # Mass (kg)
c = 0.52 * l_t  # position of COG (m)
S = 34  # Surface area (m^2)
r_g = 2.65  # Radius of gyration  (m)
z = 12800  # Altitude (ft)
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
V_sound = 324.975  # m/s
rho = 0.827683  # kg/m^3
g0 = 9.81
X_F = f * l_t
X_G = c * l_t
F_delta = f_delta * l_t
X = X_F - X_G
Y = F_delta - X_G
tau = 0.71  #time constant (s) -> used in TF for altitude

# -------- Algorithm for computing the equilibrium points --------
# Initialization
V_eq = M * V_sound
Q = 0.5 * rho * V_eq**2
alpha_eq = 0
F_p_xeq = 0
epsilon = 0.1  # desired precision in degrees
epsilon = np.radians(epsilon)

while (True):
    alpha_eq_old = alpha_eq

    C_z_eq = 1 / (Q * S) * (m * g0 - F_p_xeq * np.sin(alpha_eq))
    C_x_eq = C_x0 + k * C_z_eq**2
    F_p_xeq = (Q * S * C_x_eq) / (np.cos(alpha_eq))

    C_x_delta = 2 * k * C_z_eq * C_z_delta

    delta_m_eq = delta_m0 - (
        C_x_eq * np.sin(alpha_eq) + C_z_eq * np.cos(alpha_eq)) / (
            C_x_delta * np.sin(alpha_eq) + C_z_delta * np.cos(alpha_eq)) * (
                X / (Y - X))

    alpha_eq = alpha_0 + C_z_eq / C_z_alpha - (
        C_z_delta / C_z_alpha) * delta_m_eq
    if np.abs(alpha_eq - alpha_eq_old) < epsilon:
        break

print(
    f'alpha eq = {round(alpha_eq, 3)} radians \nF_p_eq = {round(F_p_xeq, 3)} Newtons'
)

gamma_eq = 0
Z_gamma = 0
C_x_alpha = C_x0 + k * C_z_alpha**2
C_m_delta = (Y / l_ref) * (
    C_x_delta * np.sin(alpha_eq) + C_z_delta * np.cos(alpha_eq))
C_m_alpha = (X / l_ref) * (
    C_x_alpha * np.sin(alpha_eq) + C_z_alpha * np.cos(alpha_eq))
I_yy = m * r_g**2

X_V = (2 * Q * S * C_x_eq) / (m * V_eq)
X_alpha = F_p_xeq / (m * V_eq) * np.sin(alpha_eq) + (Q * S * C_x_alpha) / (
    m * V_eq)
X_gamma = (g0 * np.cos(gamma_eq)) / V_eq
X_delta = (Q * S * C_x_delta) / (m * V_eq)

m_v = 0
m_alpha = (Q * S * l_ref * C_m_alpha) / I_yy
m_q = (Q * S * l_ref**2 * Cm_q) / (V_eq * I_yy)
m_delta = (Q * S * l_ref * C_m_delta) / I_yy

Z_V = (2 * g0) / V_eq
Z_alpha = (F_p_xeq * np.cos(alpha_eq)) / (m * V_eq) + (Q * S * C_z_alpha) / (
    m * V_eq)
Z_gamma = (g0 * np.sin(gamma_eq)) / V_eq
Z_delta_m = (Q * S * C_z_delta) / (m * V_eq)

X = np.array([V_eq, gamma_eq, alpha_eq, Q, delta_m_eq, tau]).T
A = np.array([[-X_V, -X_gamma, -X_alpha, 0, 0, 0], [Z_V, 0, Z_alpha, 0, 0, 0],
              [-Z_V, 0, -Z_alpha, 1, 0, 0], [0, 0, m_alpha, m_q, 0, 0],
              [0, 0, 0, 1, 0, 0], [0, V_eq, 0, 0, 0, 0]])
B = np.array([[0], [Z_delta_m], [-Z_delta_m], [m_delta], [0], [0]])
U = np.array([delta_m_eq])
C = np.eye(6)
D = np.zeros((6, 1))

print('A =\n', A)
print('B =\n', B)
print('C =\n', C)
print('D =\n', D)

#fig, (ax1, ax2) = plt.subplots(2, 1)
# ------ Phugoid mode ---------
A_phu = A[0:2, 0:2]
B_phu = B[0:2, 0:1]
C_phu = np.eye(2)
D_phu = np.zeros((2, 1))

sys_phu = ss(A_phu, B_phu, C_phu, D_phu)
control.matlab.damp(sys_phu)

# transfer function
sys_phu_tf = tf(sys_phu)
print("TF(Phu) = ", sys_phu_tf)

# step response
Yq, Tq = control.matlab.step(sys_phu, np.arange(0, 500, 0.01))
#ax1.plot(Tq, Yq, 'b', lw=2)
#ax1.set_title('Reponse indicielle q/Dm ')

# feedback
#Tqbo = feedback(sys_phu, 1.0)


# ------ Short Period mode ---------
A_sp = A[2:4, 2:4]
B_sp = B[2:4, 0:1]
C_sp = np.eye(2)
D_sp = np.zeros((2, 1))

sys_sp = ss(A_sp, B_sp, C_sp, D_sp)
control.matlab.damp(sys_sp)


# transfer function
sys_sp_tf = tf(sys_sp)
print("TF(SP) = ", sys_sp_tf)

# step response
Yq, Tq = control.matlab.step(sys_sp, np.arange(0,10,0.01))
#ax2.plot(Tq, Yq, 'b', lw=2)
#ax2.set_title('Reponse indicielle q/Dm ')
#plt.show()
# feedback
#Tqbo = feedback(sys_sp, 1.0)

#state vector : (gamma alpha q theta z)
A5 = A[1:,1:]
B5 = B[1:,:]
Cq = np.matrix([[0, 0, 1, 0, 0]])
Dq = np.zeros((1, 1))
sys5 = ss(A5, B5, Cq, Dq)
print(tf(sys5))
control.matlab.damp(sys5)
#sisotool(-sys5, 0.01, 1)

Kr = - 0.07

print(sys5)
#syscl = control.feedback(control.tf(Kr,1), control.tf(sys5))
syscl = control.feedback(Kr*sys5,1)
print(syscl)
print(tf(syscl))
control.matlab.damp(syscl)
Yq, Tq = control.matlab.step(syscl)
plt.figure(3)
plt.plot(Tq, Yq)
plt.show()

