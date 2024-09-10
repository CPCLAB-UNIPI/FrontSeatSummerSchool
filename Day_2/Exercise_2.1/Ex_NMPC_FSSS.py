# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:11:54 2016

@author: marcovaccari
"""
from __future__ import division
from past.utils import old_div
from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import numpy as np
from Utilities import*

### 1) Simulation Fundamentals

# 1.1) Simulation discretization parameters
Nsim = 40 # Simulation length

N = 50    # Horizon

h = 0.10 # Time step

# 3.1.2) Symbolic variables
xp = SX.sym("xp", 3) # process state vector       
x = SX.sym("x", 3)  # model state vector          
u = SX.sym("u", 2)  # control vector              
y = SX.sym("y", 2)  # measured output vector      
d = SX.sym("d", 0)  # disturbance        
# d = SX.sym("d", 2)  # disturbance               

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 2) Process and Model construction 

# 2.1) Process Parameters


# State map
def User_fxp_Cont(x,t,u,pxp,pxmp):
    """
    SUMMARY:
    It constructs the function fx_p for the non-linear case
        
    SYNTAX:
    assignment = User_fxp_Cont(x,t,u)
        
    ARGUMENTS:
    + x         - State variable     
    + t         - Current time
    + u         - Input variable  
        
    OUTPUTS:
    + fx_p      - Non-linear plant function
    """ 
    
    F0 = 0.1 #
    # F0 = if_else(t <= 5, 0.1, if_else(t<= 15, 0.11, if_else(t<= 25, 0.08, 0.1)))
    T0 = 350  # K
    c0 = 1.0  # kmol/m^3
    r = 0.219 # m
    k0 = 7.2e10 # min^-1
    EoR = 8750 # K
    U0 = 915.6*60/1000  # kJ/min*m^2*K
    rho = 1000.0 # kg/m^3
    Cp = 0.239 # kJ/kg
    DH = -5.0e4 # kJ/kmol
    Ar = math.pi*(r**2)
    
    fx_p = vertcat\
    (\
    F0*(c0 - x[0])/(Ar *x[2]) - k0*exp(-EoR/x[1])*x[0], \
    F0*(T0 - x[1])/(Ar *x[2]) -DH/(rho*Cp)*k0*exp(-EoR/x[1])*x[0] + \
    2*U0/(r*rho*Cp)*(u[0] - x[1]), \
    ((F0 - u[1])/Ar)\
    )    
    
    return fx_p

Mx = 10 # Number of elements in each time step 

# Output map
def User_fyp(x,u,t,pyp,pymp):
    """
    SUMMARY:
    It constructs the function User_fyp for the non-linear case
    
    SYNTAX:
    assignment = User_fyp(x,t)
  
    ARGUMENTS:
    + x             - State variable
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fy_p      - Non-linear plant function     
    """ 
    
    fy_p = vertcat\
    (\
    x[0],\
    x[2] \
    )
    
    return fy_p

# White Noise
#R_wn = 1e-8*np.array([[1.0, 0.0], [0.0, 1.0]]) # Output white noise covariance matrix


# 2.2) Model Parameters
    
# State Map
def User_fxm_Cont(x,u,d,t,px):
    """
    SUMMARY:
    It constructs the function fx_model for the non-linear case
    
    SYNTAX:
    assignment = User_fxm_Cont(x,u,d,t)
  
    ARGUMENTS:
    + x,u,d         - State, input and disturbance variable
    + t             - Variable that indicate the real time
    
    OUTPUTS:
    + x_model       - Non-linear model function     
    """ 
    F0 = 0.1# #TODO: what about putting it here? 
    T0 = 350  # K
    c0 = 1.0  # kmol/m^3
    r = 0.219 # m
    k0 = 7.2e10 # min^-1
    EoR = 8750 # K
    U0 = 915.6*60/1000  # kJ/min*m^2*K
    rho = 1000.0 # kg/m^3
    Cp = 0.239 # kJ/kg
    DH = -5.0e4 # kJ/kmol
    Ar = math.pi*(r**2)
    

    x_model = vertcat\
    (\
    F0*(c0 - x[0])/(Ar *x[2]) - k0*exp(-EoR/x[1])*x[0], \
    F0*(T0 - x[1])/(Ar *x[2]) -DH/(rho*Cp)*k0*exp(-EoR/x[1])*x[0] + \
    2*U0/(r*rho*Cp)*(u[0] - x[1]), \
    ((F0 - u[1])/Ar)\
    )
    
    return x_model

# Output Map
def User_fym(x,u,d,t,py):
    """
    SUMMARY:
    It constructs the function fy_m for the non-linear case
    
    SYNTAX:
    assignment = User_fym(x,u,d,t)
  
    ARGUMENTS:
    + x,d           - State and disturbance variable
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fy_p      - Non-linear plant function     
    """ 
    
    fy_model = vertcat\
                (\
                x[0],\
                x[2]\
                )
    
    return fy_model
    
Mx = 10 # Number of elements in each time step 

# 2.3) Disturbance model for Offset-free control
# offree = "nl"
offree = "no" 
# TODO: this is also to be changed if you wnat to activate the offsetfree potentiality

# 2.4) Initial condition
xs_CSTR = np.array([0.878, 324.5, 0.659])
us_CSTR = np.array([300, 0.1])
x0_m = np.array([0.05, 0.75, 0.85]) * xs_CSTR.ravel()
x0_p = np.array([0.05, 0.75, 0.85]) * xs_CSTR.ravel()
u0 = np.array([300.157, 0.1])
#dhat0 = np.array([0, 0.1]) 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 3) State Estimation
#### Extended Kalman filter tuning params ###################################
ekf = True # Set True if you want the Kalman filter
nx = x.size1()
ny = y.size1()
nd = d.size1()
Qx_kf = 1.0e-10*np.eye(nx)
Qd_kf = 1.0*np.eye(nd)
Q_kf = scla.block_diag(Qx_kf, Qd_kf)
R_kf = 1.0e-4*np.eye(ny)
P0 = 1.0*np.eye(nx+nd)  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 4) Steady-state and dynamic optimizers

# 4.1) Setpoints
def defSP(t):
    """
    SUMMARY:
    It constructs the setpoints vectors for the steady-state optimisation 
    
    SYNTAX:
    assignment = defSP(t)
  
    ARGUMENTS:
    + t             - Variable that indicates the current time
    
    OUTPUTS:
    + ysp, usp, xsp - Input, output and state setpoint values      
    """ 
    xsp = np.array([0.878, 324.5, 0.659]) # State setpoints  
    ysp = np.array([0.878, 0.659]) # Output setpoint
    usp = np.array([300, 0.1]) # Control setpoints

    return [ysp, usp, xsp]
    
# 4.2) Bounds constraints
## Input bounds
umin = np.array([295, 0.00])
umax = np.array([305, 0.25])

## State bounds
xmin = np.array([0.0, 315, 0.50])
xmax = np.array([1.0, 375, 0.75])

## Output bounds
ymin = np.array([0.0, 0.5])
ymax = np.array([1.0, 1.0])

## Disturbance bounds
dmin = -100*np.ones((d.size1(),1))
dmax = 100*np.ones((d.size1(),1))

# 4.3) Steady-state optimization : objective function
Qss = np.array([[10.0, 0.0], [0.0, 1.0]]) #Output matrix
Rss = np.array([[0.0, 0.0], [0.0, 0.0]]) # Control matrix

# 4.4) Dynamic optimization : objective function 
Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
R = np.array([[0.1, 0.0], [0.0, 0.1]])

# slacks = True

# Ws = np.eye(4)


pathfigure = 'MPC_Images/'

# def User_fobj_Coll(x,u,y,xs,us,ys,s_Coll):
#     """
#     SUMMARY:
#     It constructs the objective function for dynamic optimization

 

#     SYNTAX:
#     assignment = User_fobj_Coll(x,u,y,xs,us,ys,s_Coll)

 

#     ARGUMENTS:
#     + x,u,y         - State, input and output variables
#     + xs,us,ys      - State, input and output stationary variables
#     + s_Coll        - Internal state variables

 

#     OUTPUTS:
#     + obj         - Objective function
#     """    

#     Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#     R = np.array([[0.1, 0.0], [0.0, 0.1]])
        
#     obj = 1/N*(xQx(x,Q) + xQx(u,R))

#     return obj    

# Collocation = True