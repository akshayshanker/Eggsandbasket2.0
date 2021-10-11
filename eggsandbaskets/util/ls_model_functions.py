 

# Import packages

import numpy as np

from quantecon import tauchen
from quantecon.optimize.root_finding import brentq


import matplotlib.pyplot as plt
from itertools import product
from numba import njit, prange, jit, vectorize

from sklearn.utils.extmath import cartesian 


from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
import time
import pandas as pd
from pathos.pools import ProcessPool 
#from pathos.multiprocessing import Pool as ProcessPool
import copy



def lsmodel_function_factory(parameters, 
                                    lambdas, 
                                    normalisation):   

    #alpha_housing = parameters['alpha_housing']
    rho           = parameters['rho']
    gamma         = parameters['gamma']
    theta         = np.exp(parameters['theta'])
    k             = parameters['k']
    f_ACF         = parameters['f_ACF']         # Average contribution factor
    f_ASF         = parameters['f_ASF']         # Average service factor 
    f_LSF         = parameters['f_LSF']          # Lump sum factor 
    tzero         = parameters['tzero']
    R             = parameters['R']
    T             = parameters['T']
    r_m           = parameters['r_l']*parameters['beta_m']
    H_min = parameters['H_min']
    C_min = parameters['H_min']
    H_max = parameters['H_max']

    # adjustment cost parameters 

    nu_v_0          =        parameters['nu_v_0']
    nu_v_1          =        parameters['nu_v_1']
    nu_v_2          =        parameters['nu_v_2']
    nu_v_3          =        parameters['nu_v_3']
    nu_p_0          =        parameters['nu_p_0']
    nu_p_1          =        parameters['nu_p_1']
    nu_p_2          =        parameters['nu_p_2']
    nu_r_0          =        parameters['nu_r_0']
    nu_r_1          =        parameters['nu_r_1']
    nu_r_2          =        parameters['nu_r_2']
    nu_r_3          =        parameters['nu_r_3']
    nu_r_4          =        parameters['nu_r_4']
    psi_adj         =        parameters['psi']

    @njit
    def u(c,s,alpha):
        c = c
        s = s
        U = (((1-alpha)*(c**rho) + alpha*(s**rho))**((1-gamma)/rho) - 1)/(1-gamma) 
        return U #Verified

    @njit
    def uc(c,s, alpha):
        "Derivative of utility function wrt to consumption"

        return ((1-alpha)*(c**(rho-1)))*((1-alpha)*(c**rho) + alpha*(s**rho))**(((1-gamma)/rho)-1)

    @njit
    def ucnz(c,s, alpha):
        "Derivative of utility function wrt to consumption"

        return max(1e-200,((1-alpha)*(c**(rho-1)))*((1-alpha)*(c**rho) + alpha*(s**rho))**(((1-gamma)/rho)-1))

    @njit
    def p_power(x,y):
        return np.power(x, y)

    @njit
    def uc_vec(c,s, alpha):
        U_cvec = ((1-alpha)*(p_power(c,(rho-1))))\
                *p_power(((1-alpha)*p_power(c,rho) + alpha*p_power(s,rho)),(((1-gamma)/rho)-1))

        return U_cvec

    @njit
    def u_vec(c,s, alpha):
        c = c*1E5
        s = s*1E5

        return  (p_power(((1-alpha)*p_power(c,rho) + alpha*p_power(s,rho)),((1-gamma)/rho)) - 1)/(1-gamma) 
    
    @njit(error_model="numpy") 
    def uh(c,s, alpha):
        "Derivative of utility function wrt to housing"
        return max(1e-200,(alpha*s**(rho-1))*(((1-alpha)*(c**rho) + (alpha*s**rho)))**(((1-gamma)/rho) - 1))

    @njit
    def ces_c1(c,s,alpha,uc):
        U_c1 = (((1-alpha)*(c**(rho-1)))*((1-alpha)*(c**rho) + alpha*(s**rho))**(((1-gamma)/rho)-1)) - uc
        return U_c1  #Semi-Verified

    @njit
    def uc_inv(uc, h, alpha):
        "Inverse of MUC holding current period housing fixed"

        exp_h = alpha*(gamma-1)/(gamma*(alpha-1)-alpha)
        exp_uc = 1/(gamma*(alpha-1)-alpha)
        return max(1e-100,((uc/(1-alpha))**(exp_uc)))*max(1e-100,(h**exp_h))


    @njit
    def uh_inv(uc, h, alpha):
        "Inverse of MUH holding current period housing fixed"

        exp_h = (1-alpha*(1-gamma))/((1-alpha)*(1-gamma))
        exp_uc = 1/(1-alpha)*(1-gamma)
        return max((uc/alpha)**(exp_uc), 1e-100)*max(h**exp_h, 1e-100)
    
    @njit
    def b(A):

        "Bequest function"
        return normalisation[0]*(theta*(np.power(k+A,(1-gamma)))/(1-gamma))
    @njit 
    def b_prime(A):
        "Derivative of bequest function"
        return  normalisation[0]*theta*((k + A)**(-gamma))

    @njit
    def y(t,xi):
        if t<=R:

            tau = t - tzero 

            wage_age = np.dot(np.array([1, t, np.power(t,2), np.power(t,3), np.power(t,4)]).astype(np.float64), lambdas[0:5])
            wage_tenure = np.dot(np.array([tau, np.power(tau, 2)]), lambdas[5:8])

            return np.exp(wage_age + wage_tenure + xi)*normalisation[1] 
        else:
            return 0


    @njit
    def yvec(t,xi):

        tau = t - tzero 


        wage_age = np.dot(np.array([1, t, np.power(t,2), np.power(t,3), np.power(t,4)]).astype(np.float64), lambdas[0:5])
        wage_tenure = np.dot(np.array([tau, np.power(tau, 2)]), lambdas[5:8]) 


        return np.exp(wage_age + wage_tenure + xi)*normalisation[1]

    @njit
    def DB_benefit(t,                               # Age
               tau,                                 # Tenure 
               w,                                   # Current wage 
               Xi_index,                            # Current period shock index
               P,                                   # Probability matrix of income shocks 
               P_stat,                              # Stationary distribution 
               E,                                   # State space of income shocks 
               ):
        
        sum_1 = (1/P_stat[Xi_index])*np.sum(\
                        np.multiply(P_stat,P[:,Xi_index])*np.exp(\
                        np.dot(np.array([1, t, np.power(t-1,2), np.power(t-1,3), np.power(t-1,4)]).astype(np.float64), lambdas[0:5])\
                        + np.dot(np.array([tau-1, np.power(tau-1, 2)]).astype(np.float64), lambdas[5:8])
                        + E))

        sum_2 = (1/P_stat[Xi_index])*np.sum(\
                        np.multiply(P_stat,P[:,Xi_index])*np.exp(\
                        np.dot(np.array([1, t, np.power(t-2,2), np.power(t-2,3), np.power(t-2,4)]).astype(np.float64), lambdas[0:5])\
                        + np.dot(np.array([tau-2, np.power(tau-2, 2)]).astype(np.float64), lambdas[5:8])
                        + E))

        return f_ACF*f_ASF*f_LSF*tau*np.mean(np.array([w/normalisation[1],sum_1, sum_2]))*normalisation[1]


    @njit
    def adj_p(t_zero):
        """Gives adjustment cost at tzero for plan switching
        """
        return min(1e300,(np.exp(psi_adj) * np.exp(-500*nu_p_0 + nu_p_1*t_zero + nu_p_2*np.power(t_zero,2))))

    @njit
    def adj_v(t, a):
        """Gives adjustment cost at tzero for voluntary cont switching
        """
        var_1   = np.log(1000*a)
        cost = np.exp(psi_adj) * np.exp(nu_v_0 + nu_v_2*np.power((t - nu_v_1),2) + nu_v_3*var_1)
        cost[cost>=1e300] = 1e300
        return cost

    @njit
    def adj_pi(t, a_dc, adj_p):
        var_1 = np.log(1000*a_dc)
        cost = np.exp(psi_adj) * np.exp(nu_r_0 + nu_r_1*t + nu_r_2*np.power(t,2) + nu_r_3*var_1 + nu_r_4*adj_p)
        cost[cost>=1e300] = 1e300
        return cost 

    @njit 
    def amort_rate(t):
        return (r_m*(1+r_m)**(T-t))/((1+r_m)**(T-t)-1)

    @njit 
    def housing_ser(c, alpha, P_r):
        """Housing services as a functiin of consumption"""
        return c*np.power((alpha/(P_r*(1-alpha))),(1/(1-rho)))
        

    @njit
    def ch_ser(h, alpha, P_r):
        """Consumption as a functiin of housing services"""
        return  h*p_power((alpha/(P_r*(1-alpha))),(1/(rho-1)))

    return u, uc, uh, b, b_prime, y,yvec, DB_benefit, adj_p, adj_v, adj_pi, uc_inv, uh_inv, amort_rate, u_vec, uc_vec,housing_ser, ch_ser, ucnz, ces_c1