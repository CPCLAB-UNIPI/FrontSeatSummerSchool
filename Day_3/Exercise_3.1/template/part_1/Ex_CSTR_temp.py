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

# inputs

# Build input sequences 

# manipulated inputs as GBN

# Output Flow rate F = U[0]=


# Coolant liquid temperature Tc = U[1] =

# disturbance inputs as RW (random-walk) or merely as steady signal
        

# Outputs


# Run Simulation with RK

# Add noise (with assigned variances)

# Build Output


#### IDENTIFICATION STAGE (Linear Models)

# I/O Orders


# IN-OUT Models:

# ARX 


# ARMAX
# choose features


# SS 
# choose features


# GETTING RESULTS (Y_id)
# IN-OUT
   
# SS



##### PLOTS

# Input


# Output

    
# Explained Variance
        
 
#### VALIDATION STAGE

# Build new input sequences 


# Run Simulation with RK

# Add noise (with assigned variances) 

# Build Output


# MODEL VALIDATION   
       
# IN-OUT Models: ARX - ARMAX
# SS


##### PLOTS

# Explained Variance


        







