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
c = 0.52  # position of COG (m)
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
X_F = - f * l_t
X_G = - c * l_t
F_delta = - f_delta * l_t
X = X_F - X_G
Y = F_delta - X_G
tau = 0.71  #time constant (s) -> used in TF for altitude

# --------
# --------

# Determine the equilibrium conditions around the chosen operating point (point 42)
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

print('\n------------------------------------------------------------------')
print(X,Y)
print(
    f'\nalpha eq = {round(np.degrees(alpha_eq), 3)} degrees \nF_p_eq = {round(F_p_xeq, 3)} Newtons'
)

# --------

# Build a small signals model: give the state space representation(A, B, C, D) around this equilibrium point
gamma_eq = 0
Z_gamma = 0
C_x_alpha = 2 * k * C_z_eq * C_z_alpha
C_m_delta = (Y / l_ref) * (
    C_x_delta * np.sin(alpha_eq) + C_z_delta * np.cos(alpha_eq))
C_m_alpha = (X / l_ref) * (
    C_x_alpha * np.sin(alpha_eq) + C_z_alpha * np.cos(alpha_eq))
I_yy = m * r_g**2

X_V = (2 * Q * S * C_x_eq) / (m * V_eq)
X_alpha = F_p_xeq / (m * V_eq) * np.sin(alpha_eq) + (Q * S * C_x_alpha) / (m * V_eq)
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

# --------
print('\n------------------------------------------------------------------')

# State space representation of the system
X = np.array([V_eq, gamma_eq, alpha_eq, Q, delta_m_eq, tau]).T
A = np.array([[-X_V, -X_gamma, -X_alpha, 0, 0, 0], [Z_V, 0, Z_alpha, 0, 0, 0],
              [-Z_V, 0, -Z_alpha, 1, 0, 0], [0, 0, m_alpha, m_q, 0, 0],
              [0, 0, 0, 1, 0, 0], [0, V_eq, 0, 0, 0, 0]])
B = np.array([[0], [Z_delta_m], [-Z_delta_m], [m_delta], [0], [0]])
U = np.array([delta_m_eq])
C = np.eye(6)
D = np.zeros((6, 1))

print('\nA =\n', A)
print('\nB =\n', B)
print('\nC =\n', C)
print('\nD =\n', D)

# --------
print('\n------------------------------------------------------------------')

#Study of open loop modes: give the values of the modes, theirdamping ratio and their proper pulsation
print('\n\nOpen loop system :')
control.matlab.damp(ss(A,B,C,D))

# -------
print('\n------------------------------------------------------------------')

# Study the transient phase of the uncontrolled aircraft (shortperiod and phugoid oscillation modes)

fig, (ax1, ax2) = plt.subplots(2, 1) # setup a plot for the step responses

# ------ Phugoid mode ---------
# State space representation
A_phu = A[0:2, 0:2]
B_phu = B[0:2, 0:1]
C_phu = np.eye(2)
D_phu = 0

sys_phu = ss(A_phu, B_phu, C_phu, D_phu)
print("\n\n\nOpen loop Phugoid mode :")
control.matlab.damp(sys_phu)

# V transfer function :
tf_v = tf(ss(A_phu, B_phu, [1, 0], D_phu))
print('\nTransfer function V : \n', tf_v)

# gamma transfer function
tf_gamma = tf(ss(A_phu, B_phu, [0, 1], D_phu))
print('Transfer function gamma : \n', tf_gamma)

# step response
Yv_phu, Tv_phu = control.matlab.step(ss(tf_v))
Ygamma_phu, Tgamma_phu = control.matlab.step(ss(tf_gamma))


plt.figure()
plt.plot(Tv_phu, Yv_phu, label='v/DM')
plt.plot(Tgamma_phu, Ygamma_phu, label='gamma/DM')
plt.xlabel("Time")
plt.legend()
plt.title('Reponse indicielle Phugoid mode')
plt.savefig("Plots/reponse_indicielle_phugoid.png", dpi=300)

# ------
print('\n------------------------------------------------------------------')

# ------ Short Period mode ---------
# State space representation
A_sp = A[2:4, 2:4]
B_sp = B[2:4, 0:1]
C_sp = np.eye(2)
D_sp = 0

sys_sp = ss(A_sp, B_sp, C_sp, D_sp)
print("\n\n\nOpen loop Short Period mode :")
control.matlab.damp(sys_sp)

# Alpha transfer function :
tf_alpha = tf(ss(A_sp, B_sp, [1, 0], D_sp))
print('\nTransfer function alpha : \n', tf_alpha)

# q transfer function
tf_q = tf(ss(A_sp, B_sp, [0, 1], D_sp))
print('Transfer function q : \n', tf_q)

# step response
Yq_sp, Tq_sp = control.matlab.step(ss(tf_q))
Yalpha_sp, Talpha_sp = control.matlab.step(ss(tf_alpha))


plt.figure()
plt.plot(Tq_sp, Yq_sp, label='q/DM')
plt.plot(Talpha_sp, Yalpha_sp, label='alpha/DM')
plt.legend()
plt.xlabel("time")
plt.title('Reponse indicielle Short Period mode')
plt.savefig("Plots/reponse_indicielle_shortperiod.png", dpi=300)




# ------
print('\n------------------------------------------------------------------')

# We will now consider that the speed is controlled with an auto-throttle which is perfect (with an instantaneous response).The speed V can be removed from the state vector
# state vector : (gamma alpha q theta z)

# State space representation
A5 = A[1:, 1:]
B5 = B[1:, :]
Cq = np.matrix([[0, 0, 1, 0, 0]])
Dq = 0 #np.zeros((5,1))
sys_autothrottle = ss(A5, B5, Cq, Dq)
print('\n Transfer Function with auto-throttle :')
print(tf(sys_autothrottle))
control.matlab.damp(sys_autothrottle)
print()


# -------
print('\n------------------------------------------------------------------')
print('------ Q feedback ---------')
# With the help of sisotool (see sisopy31.py), choose the gain Kr of q feedback loop such as the closed loop damping ratio is xi=0.65.
#sisotool(-sys_autothrottle, 0.01, 1)

# after manual tuning, we find
Kr = - 0.0859

# -------
# feedback
Aq = A5 - Kr * B5 * np.array([0, 0, 1, 0, 0]) # longitudinal autopilot diapo 79
Bq = Kr * B5

#print('\nState space representation with feedback loop :\n')

# -------

print('Transfer function of the system with feedback loop :')
Tf_feedback_Q = ss2tf(ss(Aq, Bq, np.array([0, 0, 1, 0, 0]), 0))
print(Tf_feedback_Q)

# -------

control.matlab.damp(Tf_feedback_Q)

Yq, Tq = control.matlab.step(Tf_feedback_Q)
plt.figure()
plt.plot(Tq, Yq)
plt.title("Step response with feedback loop")
plt.xlabel('Time')
plt.savefig('Plots/Q_feedback_step_response.png', dpi=300)



print('\n------------------------------------------------------------------')
print('------- Q feedback with Washout filter ------')
# Choose the time constant τ of the washout filter (τs/1+τs) allowing to have the same steady state gain for alpha with or without the q feedback loop. Plot the open loop response, the closed loop response without filter and the closed loop response with the washout filter. In the following of this study, this filter will not be taken into account.

tau = 0.8
washout_filter = tf([tau,0],[tau, 1])
print('\nwashout filter : ', washout_filter)
 
# Closed loop with washout filter
feedback_with_filter_alpha = control.series(tf(1, Kr), control.feedback(tf(Kr, 1), control.series(tf_q, washout_filter)), tf_alpha)


# Closed loop without washout filter
feedback_without_filter_alpha = control.series(tf(1, Kr), control.feedback(tf(Kr, 1), tf_q), tf_alpha)

# Open loop
open_loop_alpha = tf_alpha



temps = np.linspace(0, 6, 1000)

Y_open_loop, T_open_loop = control.step_response(ss(open_loop_alpha), T=temps)
Y_feedback, T_feedback = control.step_response(feedback_without_filter_alpha, T=temps)
Y_filter, T_filter = control.step_response(feedback_with_filter_alpha, T=temps)


plt.figure()
plt.plot(Y_open_loop, T_open_loop, label='open loop')
plt.plot(Y_feedback, T_feedback, label='closed loop')
plt.plot(Y_filter, T_filter, label='closed loop filter')
plt.legend()
plt.xlabel('Time')
plt.title("Step response with feedback loop and washout filter")
plt.savefig('Plots/Q_feedback_Washout_filter.png', dpi=300)

print('\n------------------------------------------------------------------')
print('------ Gamma feedback ---------')
#Choose the gain Kγ of this flight path angle control loop with the help of sisotool;
#Propose a first choice of a gain allowing a gain margin ≥ 7 dB and a phase margin ≥ 35° and an optimized settling time (to within a 5 % threshold).

sisotool(-Tf_feedback_Q) #, 0.01, 1

#After manual tuning, we find :
#K_gamma_test = - 0.199
#Comment : If GM = 7dB then  tr5% = 1.172s. Maybe problem with OS

#Choose a second tuning (that will be kept for going on with the study), with the following requirements:
#an overshoot D1 ≤ 5%;
#a settling time at 5% tr5% for a step response that must be optimized (meaning minimized);
#the pseudo-periodic modes must be correctly damped (ξ ≥ 0.5).
"""
K_gamma = - 0.0649 #minimized tr5% = 0.846s

#State space with V_p = 0
A_gamma, B_gamma, C_gamma, D_gamma = ssdata(sys_autothrottle_Q_feedback)

sys_gamma = ss(A_gamma,B_gamma,C_gamma,D_gamma)
sys_gamma_cl = control.feedback(K_gamma * sys_autothrottle_Q_feedback, 1)
print('\nState space representation with gamma feedback loop :\n')
print(tf2ss(sys_gamma_cl))

print('\nTransfer function of the system with gamma feedback loop :\n')
print(tf(sys_gamma_cl))

control.matlab.damp(sys_gamma_cl)


Y_gamma, T_gamma = control.matlab.step(sys_gamma_cl)


plt.figure()
plt.plot(T_gamma, Y_gamma)
plt.xlabel('Time')
plt.title("Step response with feedback loop with gamma as output")
plt.savefig('Plots/step_response_feedback_gamma.png', dpi=300)


print('\n------------------------------------------------------------------')
print('------ Z feedback ---------')
#Z feedback loop

#control.sisotool(-sys_gamma_cl, 0.01, 1)
K_z = - 0.696 #There is a problem with the OS (donc sans doute avant aussi)

A_z, B_z, C_z, D_z = ssdata(sys_gamma_cl) # ici je récupère le système d'avant pour créer le nouveau, je sais pas si c'est la bonne méthode.
print("auto throttle C=", ssdata(sys_autothrottle)[2])
print("q feedback C=", C_gamma)
print("Q and gamma feedback C=", C_z)

sys_z = ss(A_z,B_z,C_z,D_z)
sys_z_cl = control.feedback(K_z * sys_gamma_cl, 1)
print('\nState space representation with feedback loop :\n')
print(tf2ss(sys_z_cl))

print('\nTransfer function of the system with feedback loop :\n')
print(tf(sys_z_cl))
control.matlab.damp(sys_z_cl)


Y_z, T_z = control.matlab.step(sys_z_cl)


plt.figure()
plt.plot(T_z, Y_z)
plt.xlabel("Time")
plt.title("Step response with feedback loop with z as output")
plt.savefig('Plots/step_response_feedback_z.png', dpi=300)
"""
