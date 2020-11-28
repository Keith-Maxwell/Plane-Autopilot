# -*- coding: utf-8 -*-

z=26110  #alt in ft
zm=7960.37 #alt in m
Mach=1.81 
dens=0.527736
temp=236.421
vsound=308.239
lt=1.5*5.24 #m length
m=8500 #kg mass
c=52/100 #% of total length - aircraft centering
S=34 #m2 reference surface of the wings
rg=2.65 #m radius of gyration 
lref=5.24 # m reference length
Iyy=m*(rg*rg)
Cx0=0.029
Czalpha=2.2
Czdeltam=0.33
deltam0=-0.01 
alpha0=0.0108
f=0.608
fdelta=0.9
k=0.51
Cmq=-0.27
g0=9.81


from math import *
import numpy as np
from control.matlab import *
import scipy, matplotlib.pyplot as plt, control, pylab #sisotool
#from sisopy3 import *




print('--------------------------------- Début du programme ---------------------------------')



#modélisation dans le cours : logigramme p47 pour trouver le pt d'équilibre (incidence d'eq, poussée, trainée,
#portance, modèle d'état (A, B, C, D), mode phugoid (oscillation d'incidence) => insertion d'un controleur
# avec feedback (boucle gyrométrique, boucle sur la pente et boucle sur l'altitude)


Xf  = -f*lt
Xg = -c*lt
X = Xf-Xg
Xfdeltam = -fdelta*lt
Y = Xfdeltam - Xg

print('----Equilibrium----')


epsilon = 1e-4

#computation of the equilibrium point
Veq = Mach*vsound
#dynamic pressure
Q = (1.0/2.0)*dens*(Veq**2) #should be in Pa, dens in kg/m3 and Veq in m/s



alphaeq = 0
Fpxeq = 0 #poussée
#            alphaeq = []
#            alphaeq.insert(0,0)
#            alphaeq.insert(1,0)
#            Fpxeq = []
#            Fpxeq.insert(0,0)
#            Fpxeq.insert(1,0)
#alphaeq[0]=alphaeq0
#Fpxeq[0]=0

#mettre les indices i entre crochets et définir les listes

Czeq = (1/(Q*S))*(m*g0 - Fpxeq*sin(alphaeq))
Cxeq = Cx0 + k*(Czeq**2)
Cxdeltam = 2*k*Czeq*Czdeltam
deltameq = deltam0 - ((Cxeq*sin(alphaeq)+Czeq*cos(alphaeq))/(Cxdeltam*sin(alphaeq)+Czdeltam*cos(alphaeq)))*(X/(Y-X))
alphaeq_new = alpha0 + Czeq/Czalpha - (Czdeltam/Czalpha)*deltameq
Fpxeq = (Q*S*Cxeq)/(cos(alphaeq_new))

#print('Czeq   ',Czeq)
#print('Cxeq   ',Cxeq)
#print('Cxdeltam'   ,Cxdeltam)
#print('Fpxeq   ',Fpxeq) #poussée?
#print('deltameq   ',deltameq)  #incidence de la gouverne de prof eq
#print('alphaeq   ',alphaeq)  #incidence d'eq
    
#
while abs(alphaeq_new-alphaeq)>=epsilon:
    alphaeq = alphaeq_new
    Czeq = (1/(Q*S))*(m*g0 - Fpxeq*sin(alphaeq))
    Cxeq = Cx0 + k*(Czeq**2)
    Cxdeltam = 2*k*Czeq*Czdeltam
    deltameq = deltam0 - ((Cxeq*sin(alphaeq)+Czeq*cos(alphaeq))/(Cxdeltam*sin(alphaeq)+Czdeltam*cos(alphaeq)))*(X/(Y-X))
    alphaeq_new = alpha0 + Czeq/Czalpha - (Czdeltam/Czalpha)*deltameq
    Fpxeq = (Q*S*Cxeq)/(cos(alphaeq_new)) #changed for i+1
    
    print('Accuracy on the angles in rad   ',epsilon)
    print()
    print('Iteration')
    print()             
    print('Czeq   ',Czeq)
    print('Cxeq   ',Cxeq)
    print('Cxdeltam'   ,Cxdeltam)
    print('Fpxeq in N   ',Fpxeq) #poussée?
    print('deltameq in rad  ',deltameq)  #incidence de la gouverne de prof eq
    print('deltameq in deg  ',deltameq*180/pi)
    print('alphaeq in rad  ',alphaeq)  #incidence d'eq
    print('alphaeq en deg   ',alphaeq*180/pi)

Xv=(2*Q*S*Cxeq)/(m*Veq)
Cxalpha=Cxeq
Cxalpha = 2*k*Czeq*Czalpha
Xalpha=(Fpxeq/(m*Veq))*sin(alphaeq)+((Q*S*Cxalpha)/(m*Veq))
gammaeq=0
Xgamma=(g0*cos(gammaeq))/Veq
Xdeltam=(Q*S*Cxdeltam)/(m*Veq)

mv=0
Cmalpha=(X/lref)*(Cxalpha*sin(alphaeq)+Czalpha*cos(alphaeq))
malpha=(Q*S*lref*Cmalpha)/Iyy
mq=(Q*S*lref*lref*Cmq)/(Veq*Iyy)
Cmdeltam=(Y/lref)*(Cxdeltam*sin(alphaeq)+Czdeltam*cos(alphaeq))
mdeltam=(Q*S*lref*Cmdeltam)/Iyy

Zv=(2*Q*S*Czeq)/(m*Veq)
Zalpha=(Fpxeq/(m*Veq))*cos(alphaeq)+((Q*S*Czalpha)/(m*Veq))
Zgamma=(g0*sin(gammaeq))/Veq
Zdeltam=(Q*S*Czdeltam)/(m*Veq)

A=np.array([[-Xv,-Xgamma,-Xalpha,0,0,0],[Zv,0,Zalpha,0,0,0],[-Zv,0,-Zalpha,1,0,0],[0,0,malpha,mq,0,0],[0,0,0,1,0,0],[0,Veq,0,0,0,0]])
B=np.array([[0],[Zdeltam],[-Zdeltam],[mdeltam],[0],[0]])
C=np.eye(6)
D=np.array([[0],[0],[0],[0],[0],[0]])

print('A matrix',A)
print('B matrix',B)
print('C matrix',C)
print('D matrix',D)

#define state space model
S_ss=control.ss(A,B,C,D)

print('State space matrix',S_ss)

State_Space_Model=control.ss(A,B,C,D)
Tf_Model=control.ss2tf(State_Space_Model)
control.matlab.damp(State_Space_Model)
print('\nTransfert function : ',Tf_Model)



############## Phugoid and short period #####################

Ar=np.array([[-Xv, -Xgamma, -Xalpha, 0],
             [Zv, 0, Zalpha, 0],
             [-Zv, 0, -Zalpha, 1],
             [0, 0, malpha, mq]]) # A matrix reduced
Br=np.array([[0],
            [Zdeltam],
            [-Zdeltam],
            [mdeltam]]) # B matrix reduced
ranking=np.linalg.matrix_rank(control.ctrb(Ar,Br))
print('\nRank of the matrix: ',ranking)

#short period
Ai=np.array([[-Zalpha, 1],
             [malpha, mq]])
Bi=np.array([[-Zdeltam],
            [mdeltam]])
Cia=np.array([1, 0])
Ciq=np.array([0, 1])
TaDeltam_ss=control.ss(Ai,Bi,Cia,0)
TaDeltam_tf=control.ss2tf(TaDeltam_ss)
control.matlab.damp(TaDeltam_ss)
print('\nTransfer function alpha/delta_m =',TaDeltam_tf)
print('\nStatic gain of alpha/delta_m =',control.dcgain(TaDeltam_tf))

TqDeltam_ss=control.ss(Ai,Bi,Ciq,0)
TqDeltam_tf=control.ss2tf(TqDeltam_ss)
control.matlab.damp(TqDeltam_ss)
print('\nTransfer function q/delta_m =',TqDeltam_tf)
print('\nStatic gain of q/delta_m =',control.dcgain(TqDeltam_tf))

plt.figure(1)
Ta, youta=control.step_response(TaDeltam_tf, np.arange(0, 20, 0.001))
plt.step(Ta, youta)
Tq, youtq=control.step_response(TqDeltam_tf, np.arange(0, 20, 0.001))
plt.step(Tq, youtq)
plt.grid(True)
plt.title('Step responses of alpha/delta_m et q/delta_m (Short Period)')
plt.legend('alpha/delta_m','q/delta_m')
plt.show()

#phugoid period

Ap=np.array([[-Xv, -Xgamma],
             [Zv, 0]])
Bp=np.array([[0],
            [Zdeltam]])
Cpv=np.array([1, 0])
Cpg=np.array([0, 1])
TvDeltam_ss=control.ss(Ap,Bp,Cpv,0)
TvDeltam_tf=control.ss2tf(TvDeltam_ss)
control.matlab.damp(TvDeltam_ss)
print('Transfer function V/Deltam =',TvDeltam_tf)
print('Static gain of V/Deltam =',control.dcgain(TvDeltam_tf))
TgDeltam_ss=control.ss(Ap,Bp,Cpg,0)
TgDeltam_tf=control.ss2tf(TgDeltam_ss)
control.matlab.damp(TgDeltam_ss)
print('Transfer function gamma/Deltam =',TgDeltam_tf)
print('Static gain of gamma/Deltam =',control.dcgain(TgDeltam_tf))

plt.figure(2)
Tv, youtv=control.step_response(TvDeltam_tf, np.arange(0, 20, 0.001))
plt.step(Tv, youtv)
Tg, youtg=control.step_response(TgDeltam_tf, np.arange(0, 20, 0.001))
plt.step(Tg, youtg)
plt.grid(True)
plt.title('Step responses of V/delta_m et gamma/Deltam (Phugoid)')
plt.legend('V/delta_m', 'gamma/Deltam')
plt.show()

plt.figure(3)
plt.subplot(2,2,1)
print(control.pzmap(TaDeltam_tf))
plt.title('Zero pole map of alpha/Deltam')
plt.subplot(2,2,2)
print(control.pzmap(TqDeltam_tf))
plt.title('Zero pole map of q/Deltam')
plt.subplot(2,2,3)
print(control.pzmap(TvDeltam_tf))
plt.title('Zero pole map of v/Deltam')
plt.subplot(2,2,4)
print(control.pzmap(TgDeltam_tf))
plt.title('Zero pole map of gamma/Deltam')
plt.show()

###################### Controler ################################
"""
sisopy3.sisotool(-TqDm_tf)
"""
Agam0=A[1:7,1:7]
Bgam0=B[1:7,0:1]
Cq=np.array([[0,0,1,0,0]])
Kr=-0.0971                  
Ak= Agam0-Kr*Bgam0*Cq
Bk= Kr*Bgam0
Ck= Cq
Dk=np.array([[0]])
TkDeltam_ss=control.ss(Ak,Bk,Ck,Dk)
TkDeltam_tf=control.ss2tf(TkDeltam_ss)
control.matlab.damp(TkDeltam_ss)
print('\nTransfer function closed loop =',TkDeltam_tf)
print('\nStatic gain of gamma/delta_m =',control.dcgain(TkDeltam_tf))

ftbf=control.feedback(Kr*TqDeltam_tf,1)

plt.figure(4)
Tk, youtk=control.step_response(TkDeltam_tf, np.arange(0, 20, 0.001))
Tfd, youtfd=control.step_response(ftbf, np.arange(0, 20, 0.001))
plt.step(Tk, youtk)
plt.step(Tfd,youtfd)
plt.grid(True)
plt.title('Step control response for controller in closed loop')
plt.show()

"""
figure(5)
sys2=control.feedback(Kr,TqDm_tf)
alpha_qc=control.series(sys2,TaDm_tf)
Talpha_qc, youtalpha_qc = control.step_response(alpha_qc, np.arange(0, 20, 0.001))
plt.step(Talpha_qc,youtalpha_qc)
plt.grid(True)
plt.show()
"""






