# Solution of radial SE using Numerov method starting from r=0

import numpy as np
import matplotlib.pyplot as plt
from numerov_functions import NumerovSolver

#close the open plots:

plt.close('all')

#GROUND STATE

# Variables:
k=1
l = 1
n= 2*k + l
E = 3/2 + n              #Energy
W = 1
hbar = 1
m = 1
N = 100000             #Number of steps
r_0 = 0         
r_f = 2
h = (r_f - r_0)/N     #Step length

r =  np.arange(r_0, r_f, h)

# Define the potential:
V = 1/2*r**2
psi_theo = np.exp(-r**2/2)
psi_theo = psi_theo/np.linalg.norm(psi_theo)
R_theo = psi_theo*r

#Solve the problem with the Numerov Solver function:
R = NumerovSolver(E, l, V, h, r)

psi = R/r
psi[0] = psi[1]
psi = psi/np.linalg.norm(psi)
R = psi*r;


plt.figure()
plt.plot(r, R_theo, label='theoretical prediction for n,l=0,0')
plt.plot(r,R, label=f'simulation for n,l={n},{l}')
plt.title(f'$R^{{{n},{l}}}(r)$', fontsize=14)
plt.xlabel('r', fontsize=14)
plt.ylabel(f'$R^{{{n},{l}}}(r)$', fontsize=14)
plt.grid()
plt.legend()


#"""
plt.figure()
plt.plot(r,psi_theo, label=f'theoretical prediction for n,l={n},{l}')
plt.plot(r,psi, label=f'simulation for n,l={n},{l}')
plt.title(f'$\psi^{{{n},{l}}}(r)$', fontsize=14)
plt.xlabel('r', fontsize=14)
plt.ylabel(f'$\psi^{{{n},{l}}}(r)$', fontsize=14)
plt.grid()
plt.legend()
#"""

print(f'Both the theoretical and numerical solutions are normalized to 1: {np.linalg.norm(psi_theo):.3f}, {np.linalg.norm(psi):.3f}')

plt.show()

