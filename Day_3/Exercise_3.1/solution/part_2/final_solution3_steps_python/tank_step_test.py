import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv

def tank(levels,t,pump,valve):
    h1 = levels[0]
    h2 = levels[1]
    c1 = 0.08 # inlet valve coefficient
    c2 = 0.04 # tank outlet coefficient
    dhdt1 = c1 * (1.0-valve) * pump - c2 * np.sqrt(h1)
    dhdt2 = c1 * valve * pump + c2 * np.sqrt(h1) - c2 * np.sqrt(h2)
    if h1>=1.0 and dhdt1>0.0:
        dhdt1 = 0
    if h2>=1.0 and dhdt2>0.0:
        dhdt2 = 0
    dhdt = [dhdt1,dhdt2]
    return dhdt

# Initial conditions (levels)
h0 = [0,0]

# Time points to report the solution
tf = 200
t = np.linspace(0,tf,tf+1)

# Inputs that can be adjusted
pump = np.empty((tf+1))
pump[0] = 0
pump[1:51] = 0.5
pump[51:tf+1] = 0.25
valve = 0.0

# Record the solution
y = np.empty((tf+1,2))
y[0,:] = h0

# Simulate the tank step test
for i in range(tf):
    # Specify the pump and valve
    inputs = (pump[i],valve)
    # Integrate the model
    h = odeint(tank,h0,[0,1],inputs)
    # Record the result
    y[i+1,:] = h[-1,:]
    # Reset the initial condition
    h0 = h[-1,:]

# Construct and save data file
data = np.vstack((t,pump))
data = np.hstack((np.transpose(data),y))
np.savetxt('data.txt',data,delimiter=',')

# Plot results
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,y[:,0],'b-')
plt.plot(t,y[:,1],'r--')
plt.ylabel('Height (m)')
plt.legend(['h1','h2'])
plt.subplot(2,1,2)
plt.plot(t,pump,'k-')
plt.ylabel('Pump')

plt.xlabel('Time (sec)')
plt.show()


