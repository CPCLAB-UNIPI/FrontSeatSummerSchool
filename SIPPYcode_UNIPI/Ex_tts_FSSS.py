#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024

@author: rbdc
"""

## Design a Dynamic Model from the data of Two tank system (offline mode)

# Original code obtained from APMonitor <https://apmonitor.com/do/index.php/Main/LevelControl>

# credits to Karol Kis

### packages


# csv read/write
import csv

# miscellaneous operating system interfaces
import os

import numpy as np

import control.matlab as cnt

import matplotlib.pyplot as plt

from sippy import *
from sippy import functionset as fset
from sippy import functionsetSIM as fsetSIM


# get current directory
cwd = os.getcwd()


## Read data 

# file name
file_name = cwd + '/SIPPYcode_UNIPI/tts_data.csv'

# read data file .csv
Time, Y_1, Y_2, X_1, X_2, U_1, U_2, = [], [], [], [], [], [], []
with open(file_name, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    rowNr = 0
    #
    for rowp in reader:
        if rowNr > 0:
            Time.append(float(rowp[0]))
            Y_1.append(float(rowp[1]))
            Y_2.append(float(rowp[2]))
            X_1.append(float(rowp[3]))
            X_2.append(float(rowp[4]))
            U_1.append(float(rowp[5]))
            U_2.append(float(rowp[6]))
        rowNr += 1

# Turning data into array
Time = np.array([Time])
Y_1 = np.array([Y_1])
Y_2 = np.array([Y_2])
X_1 = np.array([X_1])
X_2 = np.array([X_2])
U_1 = np.array([U_1])
U_2 = np.array([U_2])

# Putting data together
Y = np.vstack([Y_1, Y_2]) 
X = np.vstack([X_1, X_2])
U = np.vstack([U_1, U_2])
m = 2; p = 2 

# Define identification/validation data sets
# Data
[_, N_data] = Y.shape
Y_id = Y[:, 0:round(N_data/2)]
Y_val = Y[:, round(N_data/2)+1:]
X_id = X[:, 0:round(N_data/2)]
X_val = X[:, round(N_data/2)+1:]
U_id = U[:, 0:round(N_data/2)]
U_val = U[:, round(N_data/2)+1:]
# Time
Ts = 1                                       # sampling time [sec]
[_, N_id] = U_id.shape
[_, N_val] = U_val.shape                                                  
T_id = np.linspace(0, (N_id-1)*Ts, N_id)
T_val = np.linspace(0, (N_val-1)*Ts, N_val)


#### MODEL IDENTIFICATION (Linear Models)


# IN-OUT Models

##orders
na_ords = [1, 1]
nb_ords = [[1, 1],[1, 1]]
nc_ords = [1,1] 
theta = [[0, 0],[0, 0]]

# ARX 
Id_ARX = system_identification(Y_id, U_id, 'ARX', centering = 'MeanVal', ARX_orders = [na_ords, nb_ords, theta])
K_arx = cnt.dcgain(Id_ARX.G)
print("Gain_ARX =", K_arx)

# ARMAX
# choose features
mode_id_ARMAX = 'OPT'       # OPT: optimization-based; ILLS: iterative linear-least-square
n_iter = 500                # iteration number

if mode_id_ARMAX == 'OPT':
    Id_ARMAX = system_identification(Y_id, U_id, 'ARMAX', centering = 'MeanVal', 
                                 ARMAX_orders = [na_ords, nb_ords, nc_ords, theta], max_iterations = n_iter, 
                                 ARMAX_mod = 'OPT', stab_cons = True, stab_marg = .8)
else:
    Id_ARMAX = system_identification(Y_id, U_id, 'ARMAX', centering = 'MeanVal', 
                                 ARMAX_orders = [na_ords, nb_ords, nc_ords, theta], max_iterations = n_iter)

# SS 
# choose features
method = 'PARSIM-K'
SS_ord = 4
mode_id_SS = 'fixed';          # IC: with Information Criterion; otherwise: fixed-order model
mode_sim_SS = 'pro'         # pro: process form; inno: innovation form
#
if mode_id_SS == 'IC':
    Id_SS = system_identification(Y_id, U_id, method, IC = 'AIC', SS_orders = [3,7])
else:
    Id_SS = system_identification(Y_id, U_id, method, SS_fixed_order = SS_ord)

eig_A = np.linalg.eigvals(Id_SS.A)
print("eig_A =", eig_A)
K_id = np.dot(np.dot(Id_SS.C, np.linalg.inv(np.eye(Id_SS.n) - Id_SS.A)), Id_SS.B)
print("Gain_SS =", K_id)


# GETTING RESULTS (Y_id)
# IN-OUT
Y_arx = Id_ARX.Yid
Y_armax = Id_ARMAX.Yid    
# SS
if mode_id_SS == 'proc':
    x_ss, Y_ss = fsetSIM.SS_lsim_process_form(Id_SS.A,Id_SS.B,Id_SS.C,Id_SS.D,U_id,Id_SS.x0)
else:
    x_ss, Y_ss = fsetSIM.SS_lsim_innovation_form(Id_SS.A,Id_SS.B,Id_SS.C,Id_SS.D,Id_SS.K,Y_id,U_id,Id_SS.x0)



##### PLOTS

# Input
plt.close('all')
plt.figure(1)
str_input = ['pump rate [0-1]', 'valve position [0-1]']

for i in range(m):  
    plt.subplot(m,1,i+1)
    plt.plot(T_id,U_id[i,:])
    plt.ylabel("Input " + str(i+1))
    plt.ylabel(str_input[i])
    plt.grid()  
    if i == 0:
        plt.xlabel("Time [s]")
        plt.title('identification')
plt.savefig('./SIPPYcode_UNIPI/inputsID_tts.png')

# Output
plt.figure(2)
str_output = ['tank 1 level [m]', 'tank 2 level [m]']
for i in range(p): 
    plt.subplot(p,1,i+1)
    plt.plot(T_id,Y_id[i,:])
    plt.plot(T_id,Y_arx[i,:])
    plt.plot(T_id,Y_armax[i,:])
    plt.plot(T_id,Y_ss[i,:])
    plt.ylabel("Output " + str(i+1))
    plt.ylabel(str_output[i])
    plt.legend(['Data','ARX','ARMAX','SS'])
    plt.grid()
    if i == 0:
        plt.xlabel("Time [s]")
        plt.title('identification')
plt.savefig('./SIPPYcode_UNIPI/outputsID_tts.png')
    
# Explained Variance
        
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
       
EV_arx = ex_var(Y_arx, Y_id)
print('Explained Variance ARX-ID',EV_arx) 
  

  
############### MODEL VALIDATION   
       
# IN-OUT Models: ARX - ARMAX
Yv_arx = fset.validation(Id_ARX, U_val, Y_val, T_val, centering = 'InitVal')
Yv_armax = fset.validation(Id_ARMAX, U_val, Y_val, T_val, centering = 'InitVal')
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
    plt.plot(T_val,U_val[i,:])
    plt.ylabel(str_input[i])
    plt.grid()
    plt.xlabel("Time")
    if i == 0:
        plt.title('validation')
plt.savefig('./SIPPYcode_UNIPI/inputsVAL_tts.png')


# Output
plt.figure(4)
for i in range(p): 
    plt.subplot(p,1,i+1)
    plt.plot(T_val,Y_val[i,:])
    plt.plot(T_val,Yv_arx[i,:])
    plt.plot(T_val,Yv_armax[i,:])
    plt.plot(T_val,Yv_ss[i,:])
    plt.ylabel(str_output[i])
    plt.legend(['Data','ARX','ARMAX','SS'])
    plt.grid()
    plt.xlabel("Time")
    if i == 0:
        plt.title('validation')
plt.savefig('./SIPPYcode_UNIPI/outputsVAL_tts.png')
         

# Explained Variance
EV_arx = ex_var(Yv_arx, Y_val)
print('Explained Variance ARX-VAL',EV_arx)

    
    