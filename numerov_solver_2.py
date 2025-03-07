# SOLUTION OF SE USING NUMEROV METHOD USING THE CLASSICAL TURNING POINT
# after the classical turning point the wavefunction has no nodes and goes to zero
# for this reason we can solve the problem before and after the turning point separately
# and we'll get a more stable solution than the one obtained in numerov_solver_1.py

import numpy as np
import matplotlib.pyplot as plt
from numerov_functions import TPNumerovSolver

#close the open plots:

plt.close('all')

#GROUND STATE

# Variables:
n=30 #Principal quantum number
l = 0 #Angular momentum quantum number
E = 3/2 + n               #Energy
W = 1
hbar = 1
m = 1
N = 1000            #Number of steps
r_0 = 0         
r_f = 10
h = (r_f - r_0)/N     #Step length

r =  np.arange(r_0, r_f, h)

# Define functions:
V = 1/2*r**2
r_inv = np.sqrt(2*E) #classical turning point
r_inv_index = np.argmin(np.abs(r-r_inv))

if(n==0):
    psi_theo = np.exp(-r**2/2)
if(n==1):
    psi_theo = np.exp(-r**2/2)*r
if(n==2):
    psi_theo = (r**2-3/2)*np.exp(-r**2/2)
if(n<=2):
    psi_theo = psi_theo/np.linalg.norm(psi_theo)
    R_theo = psi_theo*r

#Solve the problem with the Numerov Solver using the turning point:
R = TPNumerovSolver(n, l, V, h, r)


#Normalization of the WF:
psi = R/r
psi[0] = psi[1]
psi = psi/np.linalg.norm(psi)
R = psi*r;


#Plot the results:
plt.figure()
if(n==0 and l==0):
    plt.plot(r, R_theo, label='theoretical prediction for n,l=0,0')
if(n==1 and l==0):
    plt.plot(r, R_theo, label='theoretical prediction for n,l=1,0')
if(n==2 and l==0):
    plt.plot(r, R_theo, label='theoretical prediction for n,l=2,0')
plt.plot(r,R, label=f'simulation for n,l={n},{l}')
plt.title(f'$R^{{{n},{l}}}(r)$', fontsize=14)
plt.xlabel('r', fontsize=14)
plt.ylabel(f'$R^{{{n},{l}}}(r)$', fontsize=14)
plt.grid()
plt.legend()


plt.figure()
if(n<=2 and l==0):
    plt.plot(r,psi_theo, label=f'theoretical prediction for n,l={n},0')
plt.plot(r,psi, label=f'simulation for n,l={n},{l}')
plt.title(f'$\psi^{{{n},{l}}}(r)$', fontsize=14)
plt.xlabel('r', fontsize=14)
plt.ylabel(f'$\psi^{{{n},{l}}}(r)$', fontsize=14)
plt.grid()
plt.legend()

#print(f"the norm of both theoretical psi and computed are 1:", np.linalg.norm(psi_theo), np.linalg.norm(psi))

plt.show()
