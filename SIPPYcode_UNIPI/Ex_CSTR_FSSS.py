# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Riccardo Bacci di Capaci

CSTR example

A Continuous Stirred Tank to be identified from input-output data
Example 1.11 of MD&Co. (Pannocchia and Rawlings, 2003)

"""

# import package
from __future__ import division         # compatibility layer between Python 2 and Python 3
from past.utils import old_div
from sippy import functionset as fset
from sippy import functionsetSIM as fsetSIM
from sippy import functionset_OPT as fset_OPT
from sippy import *
#
#
import numpy as np
import math
import control.matlab as cnt
import matplotlib.pyplot as plt


# Irreversible, first-order reaction A -> B occurs in the liquid phase
# the reactor temperature is regulated with external cooling.

# sampling time
ts = 1                      # [min]

# time settings (t final, samples number, samples vector)
tfin = 600
npts = int(old_div(tfin,ts)) + 1
Time = np.linspace(0, tfin, npts)

# Data        
F0 = 0.1                    # m^3/min
T0 = 350                    # K
c0 = 1.0                    # kmol/m^3
Tc0 = 300                   # K
r = 0.219                   # m
k0 = 7.2e10                 # min^-1
EoR = 8750                  # K
U0 = 54.94                  # kJ/min*m^2*K
rho = 1000.0                # kg/m^3
Cp = 0.239                  # kJ/kg
DH = -5.0e4                 # kJ/kmol
pi = math.pi

# Open-Loop (very sensitive) steady state
Fs = 0.1                # m^3/min
Tcs = 300.0             # m^3/min
c_in = 0.878            # kmol/m^3
T_in = 324.5            # K
h_in = 0.659            # m

# Output Initial conditions (Open-Loop steady state)
c_in = 0.878            # kmol/m^3
T_in = 324.5            # K
h_in = 0.659            # m

# VARIABLES

# 3 Inputs
# - as v. manipulated
# Output Flow rate F 
# Coolant liquid temperature Tc
# - as disturbances
# Input Flow rate F0 

# U = [F, Tc, F0]
m = 3   

# 3 States
# Output Concentration c          
# Output Temperature T    
# Level h                
# X = [c, T, h]

# 2 (Controlled) Outputs
# Y = [c, h]
p = 3 


# Function with Nonlinear System Dynamics
def Fdyn(X,U):
    
    # Mass Balance (of A)
    # dc/dt = (F0*c0 - F*c)/pi*r^2*h - k0*exp(-E/RT)*c 
    # dc/dt = F0*(c0 - c)/pi*r^2*h - k0*exp(-E/RT)*c 
    #
    dx_0 = U[2]*(c0 - X[0])/(pi*r**2*X[2]) - k0*np.exp(-EoR/X[1])*X[0]
    
    # Energy Balance
    # dT/dt = (F0*T0 - F*T)/pi*r^2*h - dHr/(rho*cp)*k0*exp(-E/RT)*c  + 2*U(Tc - T)/(rho*cp*r)
    # dT/dt = F0*(T0 - T)/pi*r^2*h - dHr/(rho*cp)*k0*exp(-E/RT)*c  + 2*U(Tc - T)/(rho*cp*r)
    #
    dx_1 = U[2]*(T0 - X[1])/(pi*r**2*X[2]) - (DH/(rho*Cp))*k0*np.exp(-EoR/X[1])*X[0] + 2*U0*(U[1] - X[1])/(rho*Cp*r)
    
    # Level Balance
    #dh/dt = (F0 - F)/(pi*r^2)
    dx_2 = (U[2] - U[0])/(pi*r**2)
    
    # Append outputs
    fx = np.hstack((dx_0, dx_1, dx_2))
    
    return fx

# Explicit Runge-Kutta 4 (TC dynamics is integrateed by hand)    
def run_RK(X,U,npts,ts):
    for j in range(npts-1):          
        Mx = 5                  # Number of elements in each time step
        dt = ts/Mx              # integration step
        # Output & Input
        X0k = X[:,j]
        Uk = U[:,j]
        # Integrate the model
        for i in range(Mx):         
            k1 = Fdyn(X0k, Uk)
            k2 = Fdyn(X0k + dt/2.0*k1, Uk)
            k3 = Fdyn(X0k + dt/2.0*k2, Uk)
            k4 = Fdyn(X0k + dt*k3, Uk)
            Xk_1 = X0k + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        X[:,j+1] = Xk_1    
    
    return X

# Function for explained variance computation (EV = R^2 of Excel)
def ex_var(y_m,y):
    # y_m       model output
    # y         original output
    y = 1. * np.atleast_2d(y)
    [n1, n2] = y.shape
    [m1, m2] = y_m.shape
    p = min(n1, n2)
    N = max(n1, n2)
    M = max(m1, m2)
    if N == n2:
        y = y.T
    if M == m2:
        y_m = y_m.T
    EV = []
    for i in range(p):
        ev = 100*(np.round((1.0 - np.mean((y[:,i] - y_m[:,i])**2)/np.var(y[:,i])), 4))
        EV.append(ev)
    return EV  

##### COLLECT DATA
    
# Input
    
# Build input sequences 
U = np.zeros((m,npts))
# manipulated inputs as GBN

# Output Flow rate F = U[0]=
prob_switch_1 = 0.05
delta1 = 0.0005
F_min = Fs - delta1
F_max = Fs + delta1 
Range_GBN_1 = [F_min,F_max]
[U[0,:],_,_] = fset.GBN_seq(npts, prob_switch_1, Range = Range_GBN_1)    

# Coolant liquid temperature Tc = U[1] =
prob_switch_2 = 0.05
delta2 = 0.5
Tc_min = Tcs - delta2
Tc_max = Tcs + delta2
Range_GBN_2 = [Tc_min,Tc_max]
[U[1,:],_,_] = fset.GBN_seq(npts, prob_switch_2, Range = Range_GBN_2)

# disturbance inputs as RW (random-walk) or merely as steady signal
mode = 'rw'

# Input Flow rate F = U[2]
if mode == 'rw':
    sigma_F0 = 1e-5
    U[2,:] = fset.RW_seq(npts, F0, sigma = sigma_F0)
    #
else:
    U[2,:] = F0*np.ones((1,npts))
        

# Outputs

# Initial Guess
Xo1 = c_in*np.ones((1,npts))
Xo2 = T_in*np.ones((1,npts))
Xo3 = h_in*np.ones((1,npts))
X = np.vstack((Xo1,Xo2,Xo3))


# Run Simulation with RK
X = run_RK(X,U,npts,ts)

# Add noise (with assigned variances)
var = [1e-6, 1e-6, 1e-6]    
noise = fset.white_noise_var(npts,var)    

# Build Output
Y = X + noise


#### IDENTIFICATION STAGE (Linear Models)

# I/O Orders
na_ords = [2,2,1] 
nb_ords = [[1,1,1], [1,1,1], [1,1,1]]
nc_ords = [1,1,1] 
theta = [[1,1,1], [1,1,1], [1,1,1]]


# IN-OUT Models:

# ARX 
Id_ARX = system_identification(Y, U, 'ARX', centering = 'MeanVal', ARX_orders = [na_ords, nb_ords, theta])

# ARMAX
# choose features
mode_id_ARMAX = 'OPT'       # OPT: optimization-based; ILLS: iterative linear-least-square
n_iter = 500                # iteration number

if mode_id_ARMAX == 'OPT':
    Id_ARMAX = system_identification(Y, U, 'ARMAX', centering = 'MeanVal', 
                                 ARMAX_orders = [na_ords, nb_ords, nc_ords, theta], max_iterations = n_iter, 
                                 ARMAX_mod = 'OPT', stab_cons = True, stab_marg = .8)
else:
    Id_ARMAX = system_identification(Y, U, 'ARMAX', centering = 'MeanVal', 
                                 ARMAX_orders = [na_ords, nb_ords, nc_ords, theta], max_iterations = n_iter)

# SS 
# choose features
method = 'PARSIM-K'
SS_ord = 5
mode_id_SS = 'IC';          # IC: with Information Criterion; otherwise: fixed-order model
mode_sim_SS = 'pro'         # pro: process form; inno: innovation form
#
if mode_id_SS == 'IC':
    Id_SS = system_identification(Y, U, method, IC = 'AIC', SS_orders = [3,10])
else:
    Id_SS = system_identification(Y, U, method, SS_fixed_order = SS_ord)

# GETTING RESULTS (Y_id)
# IN-OUT
Y_arx = Id_ARX.Yid
Y_armax = Id_ARMAX.Yid    
# SS
if mode_id_SS == 'proc':
    x_ss, Y_ss = fsetSIM.SS_lsim_process_form(Id_SS.A,Id_SS.B,Id_SS.C,Id_SS.D,U,Id_SS.x0)
else:
    x_ss, Y_ss = fsetSIM.SS_lsim_innovation_form(Id_SS.A,Id_SS.B,Id_SS.C,Id_SS.D,Id_SS.K,Y,U,Id_SS.x0)
    


##### PLOTS

# Input
plt.close('all')
plt.figure(1)

str_input = ['F [m$^3$/min]', 'T$_c$ [K]', 'F$_0$ [m$^3$/min]']
for i in range(m):  
    plt.subplot(m,1,i+1)
    plt.plot(Time,U[i,:])
    plt.ylabel("Input " + str(i+1))
    plt.ylabel(str_input[i])
    plt.grid()
    plt.xlabel("Time")
    plt.axis([0, tfin, 0.99*np.amin(U[i,:]), 1.01*np.amax(U[i,:])])
    if i == 0:
        plt.title('identification')
plt.savefig('./SIPPYcode_UNIPI/inputsID_cstr.png')

# Output
plt.figure(2)
str_output = ['c [kmol/m$^3$]', 'T [K]', 'h [m]']
for i in range(p): 
    plt.subplot(p,1,i+1)
    plt.plot(Time,Y[i,:])
    plt.plot(Time,Y_arx[i,:])
    plt.plot(Time,Y_armax[i,:])
    plt.plot(Time,Y_ss[i,:])
    plt.ylabel("Output " + str(i+1))
    plt.ylabel(str_output[i])
    plt.legend(['Data','ARX','ARMAX','SS'])
    plt.grid()
    plt.xlabel("Time")
    if i == 0:
        plt.title('identification')
plt.savefig('./SIPPYcode_UNIPI/outputsID_cstr.png')
    
# Explained Variance
EV_arx = ex_var(Y_arx, Y)
print('Explained Variance ARX-ID',EV_arx)    
 
#### VALIDATION STAGE

# Build new input sequences 
U_val = np.zeros((m,npts))
  
# Output Flow rate F = U[0]
Range_GBN_1 = [F_min,F_max]
[U_val[0,:],_,_] = fset.GBN_seq(npts, prob_switch_1, Range = Range_GBN_1)    

# Coolant liquid temperature Tc = U[1]
prob_switch_2 = 0.05
[U_val [1,:],_,_] = fset.GBN_seq(npts, prob_switch_2, Range = Range_GBN_2)

# disturbance inputs as RW (random-walk) or merely as steady signal
# Input Flow rate F = U[2]
if mode == 'rw':
    U_val[2,:] = fset.RW_seq(npts, F0, sigma = sigma_F0)
    #
else:
    U_val[2,:] = F0*np.ones((1,npts))

#### COLLECT DATA

# Output Initial conditions
X_val = np.vstack((Xo1,Xo2,Xo3))

# Run Simulation with RK
X = run_RK(X_val,U_val,npts,ts)

# Add noise (with assigned variances)
var = [1e-6, 1e-6, 1e-6]    
noise_val = fset.white_noise_var(npts,var)    

# Build Output
Y_val = X_val + noise_val


# MODEL VALIDATION   
       
# IN-OUT Models: ARX - ARMAX
Yv_arx = fset.validation(Id_ARX, U_val, Y_val, Time, centering = 'MeanVal')
Yv_armax = fset.validation(Id_ARMAX, U_val, Y_val, Time,centering = 'MeanVal')
# SS
if mode_id_SS == 'proc':
    xv_ss, Yv_ss = fsetSIM.SS_lsim_process_form(Id_SS.A,Id_SS.B,Id_SS.C,Id_SS.D,U_val,Id_SS.x0)
else:
    xv_ss, Yv_ss = fsetSIM.SS_lsim_innovation_form(Id_SS.A,Id_SS.B,Id_SS.C,Id_SS.D,Id_SS.K,Y_val,U_val,Id_SS.x0)


##### PLOTS

# Input
plt.figure(3)
for i in range(m):  
    plt.subplot(m,1,i+1)
    plt.plot(Time,U_val[i,:])
    plt.ylabel(str_input[i])
    plt.grid()
    plt.xlabel("Time")
    plt.axis([0, tfin, 0.99*np.amin(U_val[i,:]), 1.01*np.amax(U_val[i,:])])
    if i == 0:
        plt.title('validation')
plt.savefig('./SIPPYcode_UNIPI/inputsVAL_cstr.png')

# Output
plt.figure(4)
for i in range(p): 
    plt.subplot(p,1,i+1)
    plt.plot(Time,Y_val[i,:])
    plt.plot(Time,Yv_arx[i,:])
    plt.plot(Time,Yv_armax[i,:])
    plt.plot(Time,Yv_ss[i,:])
    plt.ylabel(str_output[i])
    plt.legend(['Data','ARX','ARMAX','SS'])
    plt.grid()
    plt.xlabel("Time")
    if i == 0:
        plt.title('validation')
plt.savefig('./SIPPYcode_UNIPI/outputsVAL_cstr.png')
         

# Explained Variance
EV_arx = ex_var(Yv_arx, Y_val)
print('Explained Variance ARX-VAL',EV_arx)

        
# ## Comparing with Linearized model

# A_l = np.array([[0.2681, -0.00338, -0.00728], [9.703, 0.3279, -25.44], [0, 0, 1]])
# Bu_l = np.array([[-0.00537, 0.1655], [1.297, 97.91], [0, -6.637]])
# Bd_l = np.array([[-0.1175], [69.74] , [6.637]])
# B_l = np.hstack([Bu_l, Bd_l])
# C_l = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# D_l = np.zeros((p,m))
# x0_l = np.atleast_2d(Y_val[:,0])
# # 
# sys1 = cnt.ss(A_l, B_l, C_l, D_l, ts)
# sys2 = cnt.ss(Id_SS.A,Id_SS.B,Id_SS.C,Id_SS.D, ts)
# #
# K_1 = cnt.dcgain(sys1); K_2 = cnt.dcgain(sys2); 


# # Step Response (unitary +1)
# ys1, tstep1 = cnt.step(sys1)
# ys2, tstep2 = cnt.step(sys2)
# #
# plt.figure(5)
# plt.plot(tstep1, ys1[:,:,0]), plt.grid(),plt.plot(tstep2, ys2[:,:,0]), 
# plt.xlabel("Time")
# plt.ylabel("Output")
# plt.title("Step response")
# plt.grid()

# ## Bode Plots
# w_v = np.logspace(-3,4,num=701)
# plt.figure(6)
# mag1, fi1, om = cnt.bode(sys1,w_v)
# mag2, fi2, om = cnt.bode(sys2,w_v)
# plt.subplot(2,1,1), plt.loglog(om,mag1), plt.grid(), 
# plt.loglog(om,mag1), plt.loglog(om,mag2),
# plt.xlabel("w"),plt.ylabel("Amplitude Ratio"), plt.title("Bode Plot")
# plt.subplot(2,1,2), plt.semilogx(om,fi1), plt.grid()
# plt.semilogx(om,fi1), plt.semilogx(om,fi2), 
# plt.xlabel("w"),plt.ylabel("phase")
# plt.legend(['System', 'SYS1', 'SYS2'])


# # SS
# xv_ss, Yid_ssl = fsetSIM.SS_lsim_process_form(A_l,B_l,C_l,D_l,U,x0_l)

# # Output
# plt.figure(2)
# for i in range(p):
#     plt.subplot(p,1,i+1)
#     plt.plot(Time,Yid_ssl[i,:])
#     plt.legend(['Data','ARX','ARMAX','SS','SS_lin'])
#     plt.grid()







