#CLASSICAL HARMONIC OSCILLATOR

#This code implements the Euler and Numerov methods to solve the classical harmonic
#oscillator problem and compare the results with the analytical solution.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


#variables definition:
ti = 0 # [s] starting time
tf = 10 # [s] final time
N = 1000 # [#] number of steps
h = (tf - ti)/N # [#] time step
x0 = -1 # [m] starting position
v0 = 0 # [m/s] starting velocity
k = 10 # [N/m] elastic constant
m = 0.1 # [kg] mass of the oscillator
w = math.sqrt(k/m) # [Hz] angular frequency
A = math.sqrt(x0**2 + m*v0**2/k) #[m] amplitude
phi = math.asin(x0 / A) if A != 0 else 0

#times array definition:
t = np.array([])

for i in range(N):
    if i == 0:
        t = np.append(t, ti)
    else:    
        t = np.append(t, t[i-1]+h)


#position array definition:
x = np.array([])

#analytical solution:
for i in range(N):
    x = np.append(x, A * np.sin(w * t[i] + phi))


plt.plot(t, x, label='analytical solution')


#Euler method:
x_e = np.array([x0, x0 + v0*h])

for i in range(2, N):
    next_x = h**2 * (-(k**2)*x[i]) - x[i-1] + 2*x[i]
    x_e = np.append(x_e, next_x)

plt.plot(t, x_e, label='Euler')

err_e = x[-1] - x_e[-1]


#Numerov method:
f1 = (1 + (h*k)**2/12)
f2 = (2 - 5*(h*k)**2/6)
r = f2/f1 

x_n = np.array([x0, x0 + v0*h])

for i in range(2, N):
    x_n = np.append(x_n, r*x[i] - x[i-1])

plt.plot(t, x_n, label='Numerov')

err_n = x[-1] - x_n[-1]

print(f"errore allo step N con Numerov:", err_n, " < errore allo step N con Eulero:", err_e, " < ", N*h**2)
plt.legend(loc='upper left')
plt.show()