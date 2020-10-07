 

# Import packages

import numpy as np

from quantecon import tauchen

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
	theta         = parameters['theta']
	k             = parameters['k']
	f_ACF         = parameters['f_ACF']         # Average contribution factor
	f_ASF         = parameters['f_ASF']         # Average service factor 
	f_LSF         = parameters['f_LSF']          # Lump sum factor 
	tzero         = parameters['tzero']
	R             = parameters['R']
	T             = parameters['T']
	r_m             = parameters['r_l']*parameters['beta_m']

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
	def u(c,h, alpha_housing):
		"The utility function"
		#if c>0 and h>0:
		return ((c**((1-gamma)*(1-alpha_housing)))*(h**(alpha_housing*(1-gamma)))-1)/(1-gamma)
		#else:
		#	return -np.inf

	@njit
	def uc(c,h, alpha):
		"Derivative of utility function wrt to consumption"

		return (1-alpha)*(c**(gamma*(alpha-1) - alpha))*(h**(alpha*(1-gamma)))

	@njit
	def uc_vec(c,h, alpha):

		return (1-alpha)*(np.power(c,(gamma*(alpha-1) - alpha)))*(np.power(h,(alpha*(1-gamma)))) 

	@njit
	def u_vec(c,h, alpha):

		return ((np.power(c,((1-gamma)*(1-alpha)))*(np.power(h,(alpha*(1-gamma)))))-1)/(1-gamma) 
	
	@njit
	def uh(c,h, alpha):
		"Derivative of utility function wrt to housing"
		return alpha*(c**((1-alpha)*(1-gamma)))*(h**(alpha*(1-gamma)-1))

	@njit
	def uc_inv(uc, h, alpha):
		"Inverse of derivative of MUC holding current period housing fixed"
		exp_h = alpha*(gamma-1)/(gamma*(alpha-1)-alpha)
		exp_uc = 1/(gamma*(alpha-1)-alpha)
		return ((uc/(1-alpha))**(exp_uc))*h**exp_h

	@njit
	def uh_inv(uc, h, alpha):
		"Inverse of derivative of MUH holding current period housing fixed"
		exp_h = (1-alpha*(1-gamma))/((1-alpha)*(1-gamma))
		exp_uc = 1/(1-alpha)*(1-gamma)
		return ((uc/alpha)**(exp_uc))*h**exp_h
	
	
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
		return (psi_adj + np.exp(nu_p_0 + nu_p_1*t_zero + nu_p_2*np.power(t_zero,2)))

	@njit
	def adj_v(t, a):
		"""Gives adjustment cost at tzero for voluntary cont switching
		"""
		var_1 	= np.log(a)
		var_1[var_1<=0] = 0
		return (psi_adj + np.exp(nu_v_0 + nu_v_2*np.power((t - nu_v_1),2) + nu_v_3*var_1))/normalisation[1]

	@njit
	def adj_pi(t, a_dc, adj_p):
		var_1 	= np.log(a_dc)
		var_1[var_1<=0] = 0
		return (psi_adj + np.exp(nu_r_0 + nu_r_1*t + nu_r_2*np.power(t,2) + nu_r_3*var_1) + nu_r_4*adj_p)/normalisation[1]

	@njit 
	def amort_rate(t):
		return (r_m*(1+r_m)**(T-t))/((1+r_m)**(T-t)-1)

	return u, uc, uh, b, b_prime, y,yvec, DB_benefit, adj_p, adj_v, adj_pi, uc_inv, uh_inv, amort_rate, u_vec, uc_vec