"""
NUMEROV SOLVER FUNCTION:

Description of the function:
The function numerov_solver_function(E, l, V, h, r) solves the Schrödinger equation using the Numerov method. The function takes the following inputs:
- h: Step length
- E: Energy of the system
- l: Angular momentum quantum number
- r: Radial distance (array)
- V: Potential energy (array)


"""
import numpy as np
import matplotlib.pyplot as plt

#Create a Numerov Solver function:
def NumerovSolver(n, l, V, h, r):
    E = 3/2 + n
    #define k^2:
    k_2 = np.zeros(len(r))
    for i in range(1, len(r)):
        k_2[i] = (2*E) - (2*V[i]) - l*(l+1)/r[i]**2

    #define the R wf:
    R = np.zeros(len(k_2))

    #initial conditions:
    R[1] = h**(l+1)


    #Numerov solver:
    for i in range(len(k_2)-2):
        
        f_1 = 1 + h**2 * k_2[i+2] / 12
        f_2 = 2 - 5 * h**2 * k_2[i+1] / 6
        f_3 = 1 + h**2 * k_2[i] / 12
        R[i+2] = (R[i+1]*f_2 - R[i]*f_3)/f_1

    return R





#Create a Turning Point Numerov Solver function:
#IT IS IMPORTANT TO IMPROVE THE NORMALIZATION OF THE FUNCTIONS!! Please use higher order functions for the derivatives!!
def TPNumerovSolver(n, l, V, h, r):
    #find the inversion point, where the potential is equal to the energy:
    E = 3/2 + n
    r_inv = np.sqrt(2*E)
    r_inv_index = np.argmin(np.abs(r-r_inv))+100
    r_before = r[:r_inv_index+2] #we add 2 to include the inversion point and the previous point: this is usefull to compute the derivative at the inversion point

    #define k^2:
    k_2 = np.zeros(len(r))
    for i in range(1, len(r)):
        k_2[i] = (2*E) - (2*V[i]) - l*(l+1)/r[i]**2

    #solve the problem before the turning point:
    R_before = NumerovSolver(n, l, V[:r_inv_index+2], h, r_before) #We add 2 for the same reason as before
    
    if(n%4==2):
        R_before = - R_before

    if(n%4==3):
        R_before = - R_before

    #solve the problem after the turning point:
    #At r[-1] and r[-2] we impose that the function is zero:
    R_after = np.zeros(len(r)-r_inv_index)
    R_after[-1] = h**3
    R_after[-2] = h**3


    #then, since Numerov is a symmetric formula, we can solve the problem backwards:
    for i in range(len(R_after)-3, -3, -1):
        f_1 = 1 + h**2*k_2[r_inv_index + i + 2]/12
        f_2 = 2 - 5*h**2*k_2[r_inv_index + i + 1]/6
        f_3 = 1 + h**2*k_2[r_inv_index + i]/12
        R_after[i] = (R_after[i+1]*f_2 - R_after[i+2]*f_1)/f_3

    #before concatenating the results we need to normalize the functions.
    #the ratio of the derivative over the function must be the same at the turning point.
    # We use the five-points formula to compute the derivative at the turning point:

    der_R_before = (R_before[-4] + 8*R_before[-2] - 8*R_before[-3] - R_before[-1])/(12*h)
    der_R_after = (R_after[0] + 8*R_after[2] - 8*R_after[1] - R_after[3])/(12*h)
    
    if der_R_after * der_R_before < 0:
        R_after *= -der_R_before / der_R_after
    else:
        R_after *= der_R_before / der_R_after


    #we impose that the function is continuous at the turning point:
    R_before[-1] = R_after[1]

    #we delete the last point:
    R_before = R_before[:-1]
    R_after = R_after[1:]
    #we put the last two points to zero:
    R_after[-2] = 0
    R_after[-1] = 0
    
    

    R = np.concatenate((R_before, R_after))

    return R