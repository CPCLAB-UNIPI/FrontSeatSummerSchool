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
file_name = cwd + '/tts_data.csv'

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

# Time
Ts = 1                                       # sampling time [sec]


#### MODEL IDENTIFICATION (Linear Models)


# IN-OUT Models

##orders


# ARX 

# ARMAX
# choose features

# SS 
# choose features



# GETTING RESULTS (Y_id)
# IN-OUT



##### PLOTS

# Input


# Output

    
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
       
  

  
############### MODEL VALIDATION   
       
# IN-OUT Models: ARX - ARMAX

# SS

##### PLOTS

# Input


# Output

         
# Explained Variance
    

############### MODEL COMPARISION
    





    
    