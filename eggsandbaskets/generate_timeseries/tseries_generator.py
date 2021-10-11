"""
This module contains the functions to generate time series of agents
and moments from a solved HousingModel 
 
Functions: genprofiles
			Generates the a time series of N agents from the HousingModel class 
		   
		   housingmodel_function_factory
			Generates moments from time-series 

"""

import numpy as np
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
import pandas as pd
import time
from numba import njit
from itertools import permutations
import gc

from util.helper_funcs import sim_markov, gen_policyout_arrays
import glob 
from quantecon.optimize.root_finding import brentq

import copy
import os

def genprofiles_operator(og,
				norm = 1E5):

	"""For a LifeCycle model, returns an operator to
		generate time-series 

	Parameters
	----------
	og: class
		Housing nmodel  

	norm: float
		 
	Returns
	-------
	generate_TSDF: dataframe
	"""
	
	# unpack and define functions to be used in time-series generator 
	y = og.functions.y
	adj_p, adj_v, adj_pi = og.functions.adj_p, og.functions.adj_v,\
		og.functions.adj_pi
	amort_rate = og.functions.amort_rate
	ch_ser = og.functions.ch_ser
	# Parameters
	s = og.parameters.s
	delta_housing = og.parameters.delta_housing
	tau_housing = og.parameters.tau_housing
	def_pi = og.parameters.def_pi
	beta_bar = og.parameters.beta_bar
	alpha_bar = og.parameters.alpha_bar
	beta_m = og.parameters.beta_m
	kappa_m = og.parameters.kappa_m
	r_l = og.parameters.r_l
	r_m = beta_m*r_l
	sigma_plan = og.parameters.sigma_plan
	u = og.functions.u
	uc = og.functions.ucnz

	# Grid parameters
	DC_max = og.parameters.DC_max
	C_min, C_max = og.parameters.C_min, 1
	Q_max, A_min = og.parameters.Q_max, og.parameters.A_min
	H_min, H_max = og.parameters.H_min, og.parameters.H_max
	A_max_W = og.parameters.A_max_W
	A_max_WW = og.parameters.A_max_WW
	T, tzero, R = og.parameters.T, int(og.parameters.tzero), og.parameters.R
	v_S, v_E = og.parameters.v_S, og.parameters.v_E
	r, r_l, r_H = og.parameters.r, og.parameters.r_l, og.parameters.r_H
	r_h = og.parameters.r_h
	beta_m, kappa_m = og.parameters.beta_m, og.parameters.kappa_m
	k = og.parameters.k
	phi_d = og.parameters.phi_d
	phi_c = og.parameters.phi_c
	phi_r = og.parameters.phi_r

	adj_p, adj_v, adj_pi = og.functions.adj_p, og.functions.adj_v,\
		og.functions.adj_pi
	sigma_DC_V = og.parameters.sigma_DC_V
	sigma_DB_V = og.parameters.sigma_DB_V
	sigma_DC_pi = og.parameters.sigma_DC_pi
	sigma_DB_pi = og.parameters.sigma_DB_pi

	# Exogenous shock processes
	beta_hat, P_beta, beta_stat = og.st_grid.beta_hat,\
		og.st_grid.P_beta, og.st_grid.beta_stat
	alpha_hat, P_alpha, alpha_stat = og.st_grid.alpha_hat,\
		og.st_grid.P_alpha, og.st_grid.alpha_stat
	ces_c1 = og.functions.ces_c1

	X_r, P_r = og.st_grid.X_r, og.st_grid.P_r
	Q_shocks_r, Q_shocks_P = og.st_grid.Q_shocks_r,\
		og.st_grid.Q_shocks_P
	Q_DC_shocks, Q_DC_P = og.cart_grids.Q_DC_shocks,\
		og.cart_grids.Q_DC_P

	EBA_P, EBA_P2 = og.cart_grids.EBA_P, og.cart_grids.EBA_P2

	E, P_E, P_stat = og.st_grid.E, og.st_grid.P_E, og.st_grid.P_stat

	# Smaller, "fundamental" grids
	V, Pi, DB = og.grid1d.V, og.grid1d.Pi, og.grid1d.DB

	# Medium sized and interpolation grids
	X_cont_W = og.interp_grid.X_cont_W
	X_QH_W_TS = og.interp_grid.X_QH_W_TS
	X_QH_WRTS = og.interp_grid.X_QH_WRTS

	# Grid sizes
	grid_size_A = og.parameters.grid_size_A
	grid_size_DC = og.parameters.grid_size_DC
	grid_size_H = og.parameters.grid_size_H
	grid_size_Q = og.parameters.grid_size_Q
	grid_size_M = og.parameters.grid_size_M
	grid_size_C = og.parameters.grid_size_C
	grid_size_W = int(og.parameters.grid_size_W)
	grid_size_alpha = int(og.parameters.grid_size_alpha)
	grid_size_beta = int(og.parameters.grid_size_beta)
	grid_size_HS = og.parameters.grid_size_HS

	# Preset shapes
	all_state_shape, all_state_shape_hat, v_func_shape,\
	all_state_A_last_shape, policy_shape_nadj,\
	policy_shape_adj, policy_shape_rent, prob_v_shape, prob_pi_shape = gen_policyout_arrays(og)


	@njit
	def val_diff(dc_bal, adj,t):
		adj_at_65 = adj/(beta_bar**(65-t))
		mpc_super = .05
		super_bal = 8
		h = 8

		value_at_65_1 = u(mpc_super*(super_bal +dc_bal)/1000,h/1000,alpha_bar)/(1-beta_bar)
		value_at_65_2 = u(mpc_super*(super_bal)/1000,h/1000,alpha_bar)/(1-beta_bar)
		#print(value_at_65_1 - value_at_65_2 - adj_at_65)
		return value_at_65_1 - value_at_65_2 - adj_at_65

	@njit
	def ad_u(adj,t):
		args = (adj,t)
		root = brentq(val_diff, 1e-200, 1e100, args = args, disp=False)
		return root[0]

	@njit 
	def gen_VPi(gender,t,points,\
				Age,\
				account_ind,\
				E_ind,\
				alpha_ind,\
				beta_ind,\
				pi_ushock,\
				v_ushock,\
				policy_prob_v,\
				policy_prob_pi):

		""" Genrates voluntary contribution prob and 
			risk prob"""

		prob_v_func = policy_prob_v[Age][0, E_ind,alpha_ind,beta_ind,:]

		prob_V_vals = np.empty(len(V))
		prob_V_vals[:] = eval_linear(X_cont_W,\
								 prob_v_func,\
								 points, xto.LINEAR) 

		prob_V_vals[np.isnan(prob_V_vals)] = 0.01
		
		prob_V_vals[np.where(prob_V_vals<.01)] =  0.01
		prob_V_vals[np.where(prob_V_vals>.99)] = .99

		prob_v = prob_V_vals/np.sum(prob_V_vals)
		prob_v_tilde = np.copy(prob_v)

		# Pick a random draw for the voluntary contribution (index in the vol.cont grid)
		V_ind  = np.arange(len(V))\
					[np.searchsorted(np.cumsum(prob_v), v_ushock)]
		v = V[V_ind]
		prob_pi_func = policy_prob_pi[Age][0, E_ind,alpha_ind,beta_ind,V_ind,:]
		prob_pi_vals = np.empty(len(Pi))
		prob_pi_vals[:] = eval_linear(X_cont_W,prob_pi_func, points,  xto.LINEAR) 

		prob_pi_vals[np.isnan(prob_pi_vals)] = 0.01
		prob_Pi = prob_pi_vals/np.sum(prob_pi_vals)
		prob_pi_tilde  = np.copy(prob_Pi)

		prob_pi_vals[np.where(prob_pi_vals<.01)] = .01
		prob_pi_vals[np.where(prob_pi_vals>.99)] = .99
		prob_Pi = prob_pi_vals/np.sum(prob_pi_vals)


		Pi_ind  = np.arange(len(Pi))\
					[np.searchsorted(np.cumsum(prob_Pi), pi_ushock)]
		pi = Pi[Pi_ind]

		#v_adj_c = np.log(max(1,prob_v[0]/max(1e-100,prob_v_tilde[0])))*sigma_DC_V + adj_v(t,np.array([points[1]]))
		#pi_adj_c = np.log(max(1,prob_Pi[1]/max(1e-100,prob_pi_tilde[1])))*sigma_DC_pi+ adj_pi(t,np.array([points[0]]), adj_p(17))

		#dollar_v = ad_u(np.log(v_adj_c[0]),t)
		#dollar_pi = ad_u(pi_adj_c[0],t)
		#rint(adj_v(t,np.array([points[1]])))
		#print(dollar_v)
		dollar_v = 0
		dollar_pi = 0
		return V_ind, v, Pi_ind, pi, dollar_v, dollar_pi

	@njit
	def seriesgen(acc_ind_gen, 
					age,gender,\
					wave_length,\
					W,\
					beta_hat_ts,\
					alpha_hat_ts,\
					V_ushock_ts,\
					Pi_ushock_ts,\
					DBshock,\
					policy_c_noadj,\
					policy_etas_noadj,\
					policy_a_noadj,\
					policy_c_adj,\
					policy_h_adj,\
					policy_a_adj,\
					policy_h_rent,\
					policy_zeta,\
					policy_prob_v,\
					policy_prob_pi,\
					policy_VF):

		""" Returns a time series of one agent to age 
			in baseline year and age in baseline year + wave_len
		"""

		length  = int(age + wave_length+2) 

		# Generate sequences of shocks for this agent i 
		# remeber start year is tzero. 
		# We generate shocks such that TS_j[t] gives the value of
		# the shock at t

		TS_A,TS_A_1     = 0,0
		TS_H, TS_H_1    = 0,0
		TS_DC,TS_DC_1   = 0,0 
		TS_M, TS_M1     = 0,0
		TS_C,TS_C_1     = 0,0
		TS_V,TS_V_1     = 0,0 
		TS_PI,TS_PI_1   = 0,0
		TS_wage,TS_wage_1 = 0,0
		TS_hinv,TS_hinv_1 = 0,0
		adj_V,adj_V_1   = 0,0
		adj_pi,adj_pi_1  =0,0
		P_h = np.zeros(length+1)

		#ten_shock = np.random.normal(0, 6)
		#t_0 = min(50, max(16, age - 13.26*np.log(age) + 36 - ten_shock)) #equation from excel calibration, make hard coded 
		#t_0 = max(16, int(t_0))
		
		#tzero = min(t_0, age -1)
		#@print(tzero)
		# Generate sequence of house prices
		P_h[tzero]  = 1/((1+r_H)**(age - tzero))

		for t in np.arange(tzero, len(P_h)-1):
			P_h[t+1] = (1+r_H)*P_h[t]  

		# Initialize continuous points 
		TS_A  = max(np.exp(5.44 + .2815*25 - .002448*tzero**2)/100000, A_min) #equation from excel calibration, make hard coded 
		#TS_A = A_min
		#print(TS_A)
		TS_H  = H_min
		#TS_DC = (25527*np.exp(.0448*tzero) + np.random.normal(0, 4597*np.exp(.0872*tzero)))/100000
		TS_DC = .1
		#print(TS_DC)
		#print(tzero)
		TS_M  = 0
		# Evaluate account choice
		E_ind  = int(W[tzero])
		beta_ind = int(beta_hat_ts[tzero])
		alpha_ind = int(alpha_hat_ts[tzero])
		points_plan_choice = np.array([1,1,1, P_h[tzero], 0])
		#print('done1')
		vdcfunc = policy_VF[1,E_ind,alpha_ind,beta_ind,1,:]
		vdbfunc = policy_VF[0,E_ind,alpha_ind,beta_ind,1,:]
		#print('done2')

		V_DC = eval_linear(X_cont_W,\
							vdcfunc,\
							points_plan_choice, xto.LINEAR)
		V_DB = eval_linear(X_cont_W, \
							vdbfunc, \
							points_plan_choice, xto.LINEAR)

		#print('done')

		V_DC_scaled = ((V_DC - adj_p(age))/sigma_plan)\
						 - max(((V_DC - adj_p(age))/sigma_plan), ((V_DB/sigma_plan)))
		V_DB_scaled = ((V_DB/sigma_plan)) \
						 - max(((V_DC - adj_p(age))/sigma_plan), ((V_DB/sigma_plan)))

		Prob_DC = np.exp(V_DC_scaled)/(np.exp(V_DB_scaled)\
						+   np.exp(V_DC_scaled ) )    

		dollar_p = 0


		account_ind = np.searchsorted(np.cumsum(np.array([1-Prob_DC,\
															 Prob_DC])),\
															DBshock)
		wave_data_10 = np.zeros(21)
		wave_data_14 = np.zeros(21)

		if int(account_ind) == int(acc_ind_gen):
			for t in range(int(tzero), int(length)+1):
				if t<R:

					# Get the index for the shocks
					E_ind  = int(W[t])
					beta_ind = int(beta_hat_ts[t])
					alpha_ind = int(alpha_hat_ts[t])
					h = TS_H*(1-delta_housing)
					q = P_h[t]
					E_val = E[int(W[t])]
					alpha_val   = alpha_hat[int(alpha_hat_ts[t])]
					t_minus_one = int(min(0, t-1))
					TS_M  = (1 + r_m) * TS_M * P_h[t_minus_one] / \
								((1 - delta_housing) * q)

					# Get min payment. Recall policy functions at
					# t for mortgage are defined on (1+r_m)m(t)
					# thus TS_M is after time t mortgage interest 

					# Get continuous points
					# Recall TS_DC is after interest so this is
					# (1+DC_interest)*A_DC(t)
					points = np.array([TS_A,TS_DC,TS_H, P_h[t], TS_M])
					points_zeta = np.array([TS_A/2,TS_DC,TS_H, P_h[t], TS_M])
					
					# Get discrete choice probabilities 
					pi_ushock = Pi_ushock_ts[t]
					v_ushock  = V_ushock_ts[t]
					args = (0, E_ind, alpha_ind,\
							 beta_ind,pi_ushock, v_ushock,\
							 policy_prob_v, policy_prob_pi)
					V_ind, v, Pi_ind, pi,v_adj_c, pi_adj_c = gen_VPi(gender,t,points,t-tzero, *args)


					# Calculate wage for agent 
					TS_wage = y(t,E[E_ind])
					print(E[E_ind])

					# Next period DC assets (before returns)
					# (recall domain of policy functions from def of eval_policy_W)
					# DC_prime is DC assets at the end of t, before returns into t+1
					DC_prime = TS_DC + (v +(v_S +v_E)*account_ind)*TS_wage
					TS_DC_1 = (1+(1-pi)*r_l + pi*r_h)*DC_prime



					# Wealth for renters, non-adjusters and adjusters 
					wealth_no_adj = TS_A*(1+r) + (1-v -v_S -v_E)*TS_wage 
					wealth_rent = wealth_no_adj + P_h[t]*h*(1-TS_M) #*(1+r_m)
					wealth_adj  = wealth_no_adj + P_h[t]*h*(1-TS_M)

					# Get rent and adjustment multipler values  
					zeta_func = policy_zeta[int(t-tzero)][0,E_ind,\
														alpha_ind,\
														beta_ind,\
														V_ind,
														Pi_ind,:]

					zeta_val = eval_linear(X_cont_W,zeta_func,\
											  points_zeta, xto.LINEAR) 

					# Take note of the DC value in the points_noadj
					# Policy functions are defined on the after vol.cont 
					# decision on DC 
					points_noadj = np.array([wealth_no_adj,DC_prime,TS_H, P_h[t], TS_M]) #*(1+r_m) Should there be a 1 + r_m for mortgages here?

					eta_func =  policy_etas_noadj[t-tzero][0,E_ind,\
														alpha_ind,\
														beta_ind,\
														V_ind,\
														Pi_ind, :]
					
					eta_val = eval_linear(X_cont_W,eta_func,\
										  points_noadj, xto.LINEAR)

					# Calculate if renter 
					renter1 = zeta_val > 0 

					if renter1>0:
						
						h_rent_func = policy_h_rent[t-tzero][0, E_ind,\
														alpha_ind,\
														beta_ind,\
														Pi_ind,:] 

						hs_points = np.array([wealth_rent,DC_prime, q]) 
						H_services = max(H_min, eval_linear(X_QH_WRTS, h_rent_func, hs_points,  xto.LINEAR))

						#TS_C = phi_r*q*H_services*(1-alpha_val)/alpha_val
						TS_C = min(ch_ser(H_services, alpha_val, phi_r*q), C_max)
						TS_M_1 = 0
						TS_A_1 = min(max(A_min,wealth_rent - phi_r*q*H_services - TS_C), A_max_W)
						TS_H_1 = H_min

					elif eta_val>0:

						a_noadjust_func = policy_a_noadj[int(t-tzero)][0, E_ind,\
														alpha_ind,\
														beta_ind,\
														Pi_ind,:]
						c_noadjust_func = policy_c_noadj[int(t-tzero)][0,E_ind,\
														alpha_ind,\
														beta_ind,\
														Pi_ind,:]
						no_adj_points = np.array([max(A_min,wealth_no_adj),\
													DC_prime,h,q,TS_M])
						TS_A_1 = min(max(A_min,eval_linear(X_cont_W,a_noadjust_func,\
													no_adj_points,  xto.LINEAR)), A_max_W)
						TS_C = min(max(C_min,eval_linear(X_cont_W,c_noadjust_func,\
													no_adj_points,  xto.LINEAR)), C_max)
						extra_payment = wealth_no_adj - TS_A_1 - TS_C

						total_liability_mort = TS_M*h*q - extra_payment
						
						TS_M_1 = total_liability_mort/h*q

						TS_M_1 = min((1-phi_c),TS_M_1)
						TS_H_1 = h 

					else:

						a_prime_adj_func = policy_a_adj[t-tzero][0,E_ind,\
														alpha_ind,\
														beta_ind,\
														Pi_ind,:]
						c_prime_adj_func = policy_c_adj[t-tzero][0,E_ind,\
														alpha_ind,\
														beta_ind,\
														Pi_ind,:]
						H_adj_func = policy_h_adj[t-tzero][0,E_ind,\
														alpha_ind,\
														beta_ind,\
														Pi_ind,:]

						adj_points = np.array([DC_prime,q,TS_M,\
													wealth_adj])

						TS_C = min(max(C_min,eval_linear(X_QH_W_TS,\
													c_prime_adj_func,\
													adj_points,  xto.LINEAR)), C_max)
						TS_H_1 = min(max(H_min, eval_linear(X_QH_W_TS,\
														H_adj_func,\
														adj_points,  xto.LINEAR)), H_max)
						TS_A_1 = min(max(A_min, eval_linear(X_QH_W_TS,\
														a_prime_adj_func,\
														adj_points,  xto.LINEAR)), A_max_W)

						total_liability_mort = wealth_adj - TS_A_1 - TS_C \
										- TS_H_1*q*(1+tau_housing)
											
						#total_liability_mort = TS_M*h*q - extra_payment
						TS_M_1 = - total_liability_mort/TS_H_1*q
						TS_M_1 = min((1-phi_c),TS_M_1)
						
					TS_hinv = TS_H_1 -TS_H
					TS_PI = pi
					TS_V = v

					# If t not terminal, iterate forward 
					if t == age:
						wave_data_10 = np.array([account_ind,age, TS_A*norm,TS_M*norm, renter1,\
												TS_H_1*norm, TS_DC*norm, TS_C*norm, \
												TS_wage*norm, TS_V, TS_V*TS_wage*norm,TS_PI,\
												int(TS_hinv>0),int(TS_PI!=.7), int(TS_PI>.7), int(TS_PI>.7)*TS_PI,\
												int(TS_V>0),\
												alpha_hat_ts[age], dollar_p*1E9,v_adj_c*1E9, pi_adj_c*1E9])


					if t == age + wave_length:
					# We denote the age at wave_14 by thier age at 10 so they go in 2010 bucket
						age_wave_10 = age
						age14 = age+ wave_length
						wave_data_14 = np.array([account_ind,age_wave_10, TS_A*norm,TS_M*norm, renter1,\
												TS_H_1*norm, TS_DC*norm, TS_C*norm, \
												TS_wage*norm, TS_V,TS_V*TS_wage*norm,TS_PI,\
												int(TS_hinv>0), int(TS_PI!=.7), int(TS_PI>.7),int(TS_PI>.7)*TS_PI,
												int(TS_V>0),\
												alpha_hat_ts[age14], dollar_p*1E9,v_adj_c*1E9, pi_adj_c*1E9])
					TS_A = TS_A_1
					TS_H = TS_H_1
					TS_DC = TS_DC_1
					TS_M = TS_M_1

		return wave_data_10, wave_data_14
			
	@njit
	def generate_TS(acc_ind_gen,
						gender,
						U,N,
						policy_c_noadj,
						etas_noadj,
						policy_a_noadj,
						policy_c_adj,
						policy_h_adj,
						policy_a_adj,
						policy_h_rent,
						policy_zeta,
						policy_prob_v,
						policy_prob_pi,
						policy_VF):

		"""
		Todo
		----

		Remove reshaping of prob and v policy
		"""
		TSALL_10 = np.zeros((int((int(R)-int(tzero))*N*2),21))
		TSALL_14 = np.zeros((int((int(R)-int(tzero))*N*2),21))
		wave_length = 4

		k = int(0)
		for age in np.arange(int(tzero), int(R)):
			#print(age)
			for i in range(N):
				length = int(age + wave_length+2) 
				W = sim_markov(P_E, P_stat, U[0, age, i])
				beta_hat_ts = sim_markov(P_beta, beta_stat, U[2, age, i])
				alpha_hat_ts = sim_markov(P_alpha, alpha_stat, U[1, age, i])
				V_ushock_ts  =  U[4, age, i]
				Pi_ushock_ts =  U[3, age, i]
				DBshock = U[5, age, i,0]
				w10, w14  = seriesgen(acc_ind_gen,
									age,gender,
									  wave_length,
									  W,beta_hat_ts,
									  alpha_hat_ts,
									  V_ushock_ts,
									  Pi_ushock_ts,
									  DBshock, 
									  policy_c_noadj,
										etas_noadj,
										policy_a_noadj,
										policy_c_adj,
										policy_h_adj,
										policy_a_adj,
										policy_h_rent,
										policy_zeta,
										policy_prob_v,
										policy_prob_pi,
										policy_VF)
				#print(w10)
				TSALL_10[int(k),:] = w10
				TSALL_14[int(k),:] = w14
				
				
				k +=1

				# take the antithetic
				W = sim_markov(P_E, P_stat, 1-U[0, age, i])
				beta_hat_ts = sim_markov(P_beta, beta_stat, 1-U[2, age, i])
				alpha_hat_ts = sim_markov(P_alpha, alpha_stat, 1-U[1, age, i])
				V_ushock_ts = 1-U[4, age, i]
				Pi_ushock_ts = 1-U[3, age, i]
				DBshock = 1-U[5, age, i,0]
				w10, w14  = seriesgen(acc_ind_gen,age,gender,
									 wave_length,
									 W,
									 beta_hat_ts,
									 alpha_hat_ts,
									 V_ushock_ts,
									 Pi_ushock_ts,
									 DBshock,
										policy_c_noadj,
										etas_noadj,
										policy_a_noadj,
										policy_c_adj,
										policy_h_adj,
										policy_a_adj,
										policy_h_rent,
										policy_zeta,
										policy_prob_v,
										policy_prob_pi,
										policy_VF)
				TSALL_10[int(k),:] = w10
				TSALL_14[int(k),:] = w14
				
				k +=1

		return TSALL_10, TSALL_14

	def load_pol_array(acc_ind_gen, ID, mod_name, job_path):
		""" Unpacks a policy array from 
		saved files on scratch

		Paramters
		---------
		ID: string
			model ID

		Returns
		-------

		Todo
		----
		- Use loops to perform the unpacking below"""
		#print("Loading arrays in ts generator")
		tzero = og.parameters.tzero
		R = og.parameters.R
		numpy_vars = {}
		tzero_vars_other  = {}
		tzero = og.parameters.tzero
		
		start = time.time()

		if acc_ind_gen == 0:
			os.chdir(job_path + '/{}/'.format(mod_name +'/'+ID+ '_acc_'+str(0)))
			for np_name in glob.glob('*np[yz]'):
				start = time.time()
				numpy_vars[np_name] = dict(np.load(np_name), memmap = 'r')
				

			os.chdir(job_path + '/{}/'.format(mod_name +'/'+ID+ '_acc_'+str(1)))			
			tzero_vars_other[int(tzero)] = dict(np.load("age_{}_acc_{}_id_{}_pols.npz".format(int(tzero), 1, ID)), memmap = 'r')

			var_keys = copy.copy(list(numpy_vars.keys()))

			for keys in var_keys:
				numpy_vars[keys.split('_')[1]] = numpy_vars.pop(keys)

			#print("Laoded mmap {}".format(time.time()-start))


		if acc_ind_gen == 1:
			os.chdir(job_path + '/{}/'.format(mod_name +'/'+ID+'_acc_'+str(1)))
			for np_name in glob.glob('*np[yz]'):
				numpy_vars[np_name] = dict(np.load(np_name), memmap = 'r')
				#print("Laoded mmap {}".format(time.time()-start))

			os.chdir(job_path + '/{}/'.format(mod_name +'/'+ID+ '_acc_'+str(0)))
			tzero_vars_other[int(tzero)] = dict(np.load("age_{}_acc_{}_id_{}_pols.npz".format(int(tzero), 0, ID)), memmap = 'r')
			
			var_keys = copy.copy(list(numpy_vars.keys()))
			for keys in var_keys:
				numpy_vars[keys.split('_')[1]] = numpy_vars.pop(keys)

			#print("Laoded mmap {}".format(time.time()-start))

		policy_c_noadj= []
		etas_noadj = []
		policy_a_noadj = []
		policy_c_adj = []
		policy_h_adj = []
		policy_a_adj = []
		policy_h_rent = []
		policy_zeta = []
		policy_prob_v = []
		policy_prob_pi = []

		for Age in np.arange(int(og.parameters.tzero), int(og.parameters.R)):

				start = time.time()
				policy_c_adj.append(numpy_vars[str(int(Age))]['C_adj'])
				policy_h_adj.append(numpy_vars[str(int(Age))]['H_adj'])
				policy_a_adj.append(numpy_vars[str(int(Age))]['Aprime_adj'])
				policy_c_noadj.append(numpy_vars[str(int(Age))]['C_noadj'])
				etas_noadj.append(numpy_vars[str(int(Age))]['etas_noadj'].reshape(all_state_shape_hat).astype(np.float32))
				policy_a_noadj.append(numpy_vars[str(int(Age))]['Aprime_noadj'])
				policy_zeta.append(numpy_vars[str(int(Age))]['zeta'].reshape(all_state_shape_hat).astype(np.float32))
				policy_h_rent.append(numpy_vars[str(int(Age))]['H_rent'])
				policy_prob_v.append(numpy_vars[str(int(Age))]['prob_v'].astype(np.float32))
				policy_prob_pi.append(numpy_vars[str(int(Age))]['prob_pi'].astype(np.float32))


				if Age == og.parameters.tzero and acc_ind_gen == 0:
					policy_VF = np.concatenate((numpy_vars[str(int(Age))]['policy_VF'],\
										tzero_vars_other[int(Age)]['policy_VF']))
				if Age == og.parameters.tzero and acc_ind_gen == 1:
					policy_VF = np.concatenate((tzero_vars_other[int(Age)]['policy_VF'], numpy_vars[str(int(Age))]['policy_VF']))

				del numpy_vars[str(int(Age))]
				gc.collect()

		del numpy_vars, tzero_vars_other
		gc.collect()

		return policy_c_noadj,\
				etas_noadj,\
				policy_a_noadj,\
				policy_c_adj,\
				policy_h_adj,\
				policy_a_adj,\
				policy_h_rent,\
				policy_zeta,\
				policy_prob_v,\
				policy_prob_pi,\
				policy_VF

	def generate_TSDF(gender,U,N,ID, mod_name, job_path):

		""" Unpacks polices, generates time-series, 
			labels time-series and returns data-frame"""

		policies = load_pol_array(0,ID, mod_name, job_path)
		start = time.time()

		# Generate DB
		TSALL_10_DB, TSALL_14_DB = generate_TS(0,gender,U,N,*policies)
		del policies
		gc.collect()

		policies = load_pol_array(1, ID,mod_name, job_path)
		TSALL_10_DC, TSALL_14_DC = generate_TS(1, gender,U,N,*policies)
		TSALL_10 = np.vstack((TSALL_10_DB, TSALL_10_DC))
		TSALL_14 = np.vstack((TSALL_14_DB, TSALL_14_DC))
		idx = np.argwhere(np.all(TSALL_10[:,...] == 0, axis=1))
		TSALL_10 = np.delete(TSALL_10, idx, axis=0)
		idx_2 = np.argwhere(np.all(TSALL_14[:,...] == 0, axis=1))
		TSALL_14 = np.delete(TSALL_14, idx_2, axis=0)
		
		del policies, idx_2, idx, TSALL_10_DC,\
				 TSALL_14_DC, TSALL_10_DB,TSALL_14_DB

		TSALL_10_df = pd.DataFrame(TSALL_10)
		TSALL_14_df = pd.DataFrame(TSALL_14)

		del TSALL_10
		del TSALL_14
		gc.collect()

		col_list = list(['account_type', \
							  'Age', \
							  'wealth_fin',\
							  'mortgagebal',\
							  'renter',\
							  'wealth_real',\
							  'super_balance',\
							  'consumption', \
							  'Wages',\
							  'vol_cont', \
							  'vol_total',\
							  'risk_share', \
							  'house_adj', \
							  'nondef_share_adj',\
							  'risky_share_adj',\
							  'risky_risk_share',\
							  'vol_cont_adj', \
							  'alpha_hat', \
							  'Adjustment_cost_plan', \
							  'Adjustment_cost_v', \
							  'Adjustment_cost_pi'])

		TSALL_10_df.columns = col_list
		TSALL_14_df.columns = col_list
		
		return TSALL_10_df, TSALL_14_df

	return generate_TSDF

def gen_panel_ts(gender, og, U,N, job_path):
	
	generate_TSDF = genprofiles_operator(og)
	TSALL_10_df, TSALL_14_df = generate_TSDF(gender,U,N,og.ID, og.mod_name, job_path)
	TSALL_10_df.to_pickle(job_path + '/{}/TSALL_10_df.pkl'.format(og.mod_name +'/'+ og.ID + '_acc_0')) 
	TSALL_14_df.to_pickle(job_path + '/{}/TSALL_14_df.pkl'.format(og.mod_name +'/'+ og.ID + '_acc_0'))
	
	del generate_TSDF, TSALL_10_df, TSALL_14_df
	gc.collect()

	return None 

def gen_moments(TSALL_10_df, TSALL_14_df):

	age     = np.arange(18, 65)
	main_df = pd.DataFrame(age)

	# Age buckets are LHS open RHS closed
	# final age bucket is t = (58, 63] and hence `real' age = (59, 64]
	# first age bucket is age = (19, 24] 
	age_buckets = np.arange(19, 65,5)
	keys_vars = set(TSALL_10_df.keys())
	excludes = set(['Adjustment_cost_v', \
				 'Adjustment_cost_pi', 'alpha_hat', 'Adjustment_cost_plan'])

	TSALL_10_df.drop(excludes, axis = 1)    
	
	# Wave 10
	main_df = TSALL_10_df
	main_df['vol_cont_adj']= main_df['vol_cont_adj']
	main_df['nondef_share_adj']= main_df['nondef_share_adj']
	main_df['risky_share_adj']= main_df['risky_share_adj']
	main_df['risk_share']= main_df['risk_share']
	main_df['risky_risk_share']= main_df['risky_risk_share']
	main_df['vol_cont'] = main_df['vol_cont']

	# Adjust age so it is 'real age'
	main_df['Age'] = main_df['Age']+ 1    

	# Get means
	means = main_df.groupby(pd.cut(main_df['Age'], age_buckets)).mean().add_prefix('mean_')  
	means = means.reset_index()   

	# Get standard deviation 
	sds_all = main_df.groupby(pd.cut(main_df['Age'], \
				age_buckets)).std().add_prefix('sd_')
	sds_all = sds_all.reset_index().drop(['Age','sd_Age', \
				'sd_account_type'], axis =1)    
	sds_DB  = main_df[main_df['account_type']==0].groupby(pd.cut \
				(main_df[main_df['account_type']==0]['Age'], age_buckets)) \
				.std().add_prefix('sd_').add_suffix('DB')
	sds_DB  = sds_DB.reset_index().drop(['Age','sd_AgeDB', \
				'sd_account_typeDB'], axis =1)
	sds_DC  = main_df[main_df['account_type']==1].groupby(pd. \
				cut(main_df[main_df['account_type']==1]['Age'], age_buckets)). \
				std().add_prefix('sd_').add_suffix('DC')
	sds_DC  = sds_DC.reset_index().drop(['Age','sd_AgeDC', \
				 'sd_account_typeDC'], axis =1)

	# Correlations 
	corrlist = list(list(permutations(keys_vars.difference(excludes),2)))
	corrs_df = pd.DataFrame(means.index)  

	for corrs in corrlist:

		corrs_temp = main_df.groupby(pd.cut(main_df['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp = corrs_temp.reset_index()
		corrs_temp = np.array(corrs_temp[corrs_temp.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df = pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp)  ], axis = 1) 
		corrs_df = corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)], \
							axis=1, inplace=False)

		# for DB only
		corrs_temp_DB = main_df[main_df['account_type']==0] \
						 .groupby(pd.cut(main_df[main_df['account_type']==0]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DB = corrs_temp_DB.reset_index()
		corrs_temp_DB = np.array(corrs_temp_DB[corrs_temp_DB.columns[1]] \
						.reset_index())[:,1]
		corrs_df = pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DB)  ], axis = 1) 
		corrs_df = corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DB'], \
							axis=1, inplace=False)

		# for DC only
		corrs_temp_DC = main_df[main_df['account_type']==1] \
						 .groupby(pd.cut(main_df[main_df['account_type']==1]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DC = corrs_temp_DC.reset_index()
		corrs_temp_DC = np.array(corrs_temp_DC[corrs_temp_DC.columns[1]] \
						.reset_index())[:,1]
		corrs_df = pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DC)  ], axis = 1) 
		corrs_df =   corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DC'], \
							axis=1, inplace=False)

	corrs_df = corrs_df.add_prefix('corr_')
	moments = pd.concat([means.reset_index(), sds_all.reset_index(),\
				 corrs_df.reset_index(), sds_DB.reset_index()\
				 ,sds_DC.reset_index()  ], axis = 1)   
	
	moments_10 = moments.drop(['index'],axis = 1).add_suffix('_wave10')   

	main_df10 = main_df

	main_df10 = main_df10.add_suffix('_wave10')

	# Moments for wave 14

	main_df = TSALL_14_df
	main_df['vol_cont_adj']= main_df['vol_cont_adj']
	main_df['nondef_share_adj']= main_df['nondef_share_adj']
	main_df['risky_share_adj']= main_df['risky_share_adj']
	main_df['risk_share']= main_df['risk_share']
	main_df['risky_risk_share']= main_df['risky_risk_share']
	main_df['vol_cont']=main_df['vol_cont']

	# Adjust age so it is 'real age'=
	main_df['Age'] += 1    

	#div

	means = main_df.groupby(pd.cut(main_df['Age'], age_buckets)).mean().add_prefix('mean_')  
	means = means.reset_index() 

	# SDs
	sds_all = main_df.groupby(pd.cut(main_df['Age'], \
				age_buckets)).std().add_prefix('sd_')
	sds_all = sds_all.reset_index().drop(['Age','sd_Age', \
				'sd_account_type'], axis =1)    

	sds_DB  = main_df[main_df['account_type']==0].groupby(pd.cut \
				(main_df[main_df['account_type']==0]['Age'], age_buckets))\
				.std().add_prefix('sd_').add_suffix('DB')
	sds_DB  = sds_DB.reset_index().drop(['Age','sd_AgeDB',\
				'sd_account_typeDB'], axis =1)

	sds_DC  = main_df[main_df['account_type']==1].groupby(pd. \
				cut(main_df[main_df['account_type']==1]['Age'], age_buckets)). \
				std().add_prefix('sd_').add_suffix('DC')
	sds_DC  = sds_DC.reset_index().drop(['Age','sd_AgeDC',\
				 'sd_account_typeDC'], axis =1)

	# Correlations 
	corrlist = list(list(permutations(keys_vars.difference(excludes),2)))
	corrs_df = pd.DataFrame(means.index)  

	for corrs in corrlist:

		corrs_temp  = main_df.groupby(pd.cut(main_df['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp  = corrs_temp.reset_index()
		corrs_temp  = np.array(corrs_temp[corrs_temp.columns[1]] \
						.reset_index())[:,1]
		corrs_df    = pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp)  ], axis = 1) 
		corrs_df    =   corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)], \
							axis=1, inplace=False)
		# for DB only
		corrs_temp_DB  = main_df[main_df['account_type']==0] \
						 .groupby(pd.cut(main_df[main_df['account_type']==0]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DB  = corrs_temp_DB.reset_index()
		corrs_temp_DB  = np.array(corrs_temp_DB[corrs_temp_DB.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df    = pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DB)  ], axis = 1) 
		corrs_df    =   corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DB'], \
							axis=1, inplace=False)

		# for DC only
		corrs_temp_DC  = main_df[main_df['account_type']==1] \
						 .groupby(pd.cut(main_df[main_df['account_type']==1]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DC  = corrs_temp_DC.reset_index()
		corrs_temp_DC  = np.array(corrs_temp_DC[corrs_temp_DC.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df    = pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DC)  ], axis = 1) 
		corrs_df    =   corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DC'], \
							axis=1, inplace=False)

	corrs_df        = corrs_df.add_prefix('corr_')
	moments_14      = pd.concat([means.reset_index(), sds_all.reset_index(),\
						corrs_df.reset_index(), sds_DB.reset_index()\
						,sds_DC.reset_index()  ], axis = 1) \
						.reset_index() \
						.add_suffix('_wave14') \
						.drop(['Age_wave14', 'index_wave14', 'level_0_wave14'],
							 axis= 1)
	main_df14       = main_df
	main_df14       = main_df14.add_suffix('_wave14')
	main_horiz      = pd.concat((main_df10, main_df14), axis =1)  
	
	# Auto-correlation
	auto_corrs_df = pd.DataFrame(means.index)

	for keys in keys_vars.difference(excludes):
		corrs = list((keys+'_wave10', keys+'_wave14'))
		autocorrs_temp  = main_horiz.groupby(pd.cut(main_horiz['Age_wave10'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		autocorrs_temp  = autocorrs_temp.reset_index()
		autocorrs_temp  = np.array(autocorrs_temp[autocorrs_temp.columns[1]] \
						.reset_index())[:,1]
		auto_corrs_df   = pd.concat([auto_corrs_df.reset_index(drop = True), \
							pd.DataFrame(autocorrs_temp)  ], axis = 1) 
		auto_corrs_df   =   auto_corrs_df.set_axis([*auto_corrs_df.columns[:-1], \
							 keys+'_autocorr'], \
							 axis=1, inplace=False)

	# Voluntary contribution conditional on making a voluntary contribution
	moments_10['mean_vol_cont_c_wave10'] = moments_10['mean_vol_total_wave10']/moments_10['mean_vol_cont_adj_wave10']
	moments_14['mean_vol_cont_c_wave14'] = moments_14['mean_vol_total_wave14']/moments_14['mean_vol_cont_adj_wave14']

	moments_10['mean_vol_cont_c_wave10'] = moments_10['mean_vol_cont_c_wave10'].fillna(0)
	moments_14['mean_vol_cont_c_wave14'] = moments_14['mean_vol_cont_c_wave14'].fillna(0)

	# Risky share conditioned on choosing a risky share 
	moments_10['mean_risky_risk_share_c_wave10'] = moments_10['mean_risky_risk_share_wave10']/moments_10['mean_risky_share_adj_wave10']
	moments_14['mean_risky_risk_share_c_wave14'] = moments_14['mean_risky_risk_share_wave14']/moments_14['mean_risky_share_adj_wave14']

	moments_10['mean_risky_risk_share_c_wave10'] =  moments_10['mean_risky_risk_share_c_wave10'].fillna(0)
	moments_14['mean_risky_risk_share_c_wave14'] = moments_14['mean_risky_risk_share_c_wave14'].fillna(0)

	return pd.concat([moments_10, moments_14, auto_corrs_df], axis =1)


def sortmoments(moments_male, moments_female):
	
	empty = pd.DataFrame(np.zeros(9))

	moments_sorted = pd.concat(
						[
						moments_male['sd_consumption_wave14_male'],
						moments_female['sd_consumption_wave14_female'],
						moments_male['sd_consumption_wave10_male'],
						moments_female['sd_consumption_wave10_female'],
						moments_male['mean_account_type_wave14_male'],
						moments_female['mean_account_type_wave14_female'],
						moments_male['mean_account_type_wave10_male'],
						moments_female['mean_account_type_wave10_female'],
						moments_male['corr_super_balance_risky_share_adj_DC_wave14_male'],
						moments_male['corr_super_balance_risky_share_adj_DB_wave14_male'],
						moments_female['corr_super_balance_risky_share_adj_DC_wave14_female'],
						moments_female['corr_super_balance_risky_share_adj_DB_wave14_female'],
						moments_male['corr_super_balance_risky_share_adj_DC_wave10_male'],
						moments_male['corr_super_balance_risky_share_adj_DB_wave10_male'],
						moments_female['corr_super_balance_risky_share_adj_DC_wave10_female'],
						moments_female['corr_super_balance_risky_share_adj_DB_wave10_female'],
						moments_male['corr_super_balance_risk_share_DC_wave14_male'],
						moments_male['corr_super_balance_risk_share_DB_wave14_male'],
						moments_female['corr_super_balance_risk_share_DC_wave14_female'],
						moments_female['corr_super_balance_risk_share_DB_wave14_female'],
						moments_male['corr_super_balance_risk_share_DC_wave10_male'],
						moments_male['corr_super_balance_risk_share_DB_wave10_male'],
						moments_female['corr_super_balance_risk_share_DC_wave10_female'],
						moments_female['corr_super_balance_risk_share_DB_wave10_female'],
						moments_male['corr_super_balance_vol_cont_adj_DC_wave14_male'],
						moments_male['corr_super_balance_vol_cont_adj_DB_wave14_male'],
						moments_female['corr_super_balance_vol_cont_adj_DC_wave14_female'],
						moments_female['corr_super_balance_vol_cont_adj_DB_wave14_female'],
						moments_male['corr_super_balance_vol_cont_adj_DC_wave10_male'],
						moments_male['corr_super_balance_vol_cont_adj_DB_wave10_male'],
						moments_female['corr_super_balance_vol_cont_adj_DC_wave10_female'],
						moments_female['corr_super_balance_vol_cont_adj_DB_wave10_female'],
						moments_male['corr_vol_total_super_balance_DC_wave14_male'],
						moments_male['corr_vol_total_super_balance_DB_wave14_male'],
						moments_female['corr_vol_total_super_balance_DC_wave14_female'],
						moments_female['corr_vol_total_super_balance_DB_wave14_female'],
						moments_male['corr_vol_total_super_balance_DC_wave10_male'],
						moments_male['corr_vol_total_super_balance_DB_wave10_male'],
						moments_female['corr_vol_total_super_balance_DC_wave10_female'],
						moments_female['corr_vol_total_super_balance_DB_wave10_female'],
						moments_male['corr_consumption_wealth_real_wave14_male'],
						moments_female['corr_consumption_wealth_real_wave14_female'],
						moments_male['corr_consumption_wealth_real_wave10_male'],
						moments_female['corr_consumption_wealth_real_wave10_female'],
						moments_male['wealth_real_autocorr_male'],
						moments_female['wealth_real_autocorr_female'],
						moments_male['wealth_fin_autocorr_male'],
						moments_female['wealth_fin_autocorr_female'],
						moments_male['consumption_autocorr_male'],
						moments_female['consumption_autocorr_female'],
						moments_male['sd_risk_shareDC_wave14_male'],
						moments_male['sd_risk_shareDB_wave14_male'],
						moments_female['sd_risk_shareDC_wave14_female'],
						moments_female['sd_risk_shareDB_wave14_female'],
						moments_male['sd_risk_shareDC_wave10_male'],
						moments_male['sd_risk_shareDB_wave10_male'],
						moments_female['sd_risk_shareDC_wave10_female'],
						moments_female['sd_risk_shareDB_wave10_female'],
						moments_male['sd_vol_totalDC_wave14_male'],
						moments_male['sd_vol_totalDB_wave14_male'],
						moments_female['sd_vol_totalDC_wave14_female'],
						moments_female['sd_vol_totalDB_wave14_female'],
						moments_male['sd_vol_totalDC_wave10_male'],
						moments_male['sd_vol_totalDB_wave10_male'],
						moments_female['sd_vol_totalDC_wave10_female'],
						moments_female['sd_vol_totalDB_wave10_female'],
						moments_male['sd_super_balanceDC_wave14_male'],
						moments_male['sd_super_balanceDB_wave14_male'],
						moments_female['sd_super_balanceDC_wave14_female'],
						moments_female['sd_super_balanceDB_wave14_female'],
						moments_male['sd_super_balanceDC_wave10_male'],
						moments_male['sd_super_balanceDB_wave10_male'],
						moments_female['sd_super_balanceDC_wave10_female'],
						moments_female['sd_super_balanceDB_wave10_female'],
						moments_male['sd_wealth_real_wave14_male'],
						moments_female['sd_wealth_real_wave14_female'],
						moments_male['sd_wealth_real_wave10_male'],
						moments_female['sd_wealth_real_wave10_female'],
						moments_male['sd_wealth_fin_wave14_male'],
						moments_female['sd_wealth_fin_wave14_female'],
						moments_male['sd_wealth_fin_wave10_male'],
						moments_female['sd_wealth_fin_wave10_female'],
						moments_male['mean_vol_cont_c_wave14_male'],
						moments_female['mean_vol_cont_c_wave14_female'],
						moments_male['mean_vol_cont_c_wave10_male'],
						moments_female['mean_vol_cont_c_wave10_female'],
						moments_male['mean_risk_share_wave14_male'],
						moments_female['mean_risk_share_wave14_female'],
						moments_male['mean_risk_share_wave10_male'],
						moments_female['mean_risk_share_wave10_female'],
						moments_male['mean_vol_cont_adj_wave14_male'],
						moments_female['mean_vol_cont_adj_wave14_female'],
						moments_male['mean_vol_cont_adj_wave10_male'],
						moments_female['mean_vol_cont_adj_wave10_female'],
						moments_male['mean_vol_total_wave14_male'],
						moments_female['mean_vol_total_wave14_female'],
						moments_male['mean_vol_total_wave10_male'],
						moments_female['mean_vol_total_wave10_female'],
						moments_male['mean_wealth_real_wave14_male'],
						moments_female['mean_wealth_real_wave14_female'],
						moments_male['mean_wealth_real_wave10_male'],
						moments_female['mean_wealth_real_wave10_female'],
						moments_male['mean_wealth_fin_wave14_male'],
						moments_female['mean_wealth_fin_wave14_female'],
						moments_male['mean_wealth_fin_wave10_male'],
						moments_female['mean_wealth_fin_wave10_female'],
						moments_male['mean_super_balance_wave14_male'],
						moments_female['mean_super_balance_wave14_female'],
						moments_male['mean_super_balance_wave10_male'],
						moments_female['mean_super_balance_wave10_female'],
						moments_male['mean_consumption_wave14_male'],
						moments_female['mean_consumption_wave14_female'],
						moments_male['mean_consumption_wave10_male'], 
						moments_female['mean_consumption_wave10_female'],
						moments_male['corr_nondef_share_adj_vol_cont_adj_DC_wave14_male'],
						moments_male['corr_nondef_share_adj_vol_cont_adj_DB_wave14_male'],
						moments_female['corr_nondef_share_adj_vol_cont_adj_DC_wave14_female'],
						moments_female['corr_nondef_share_adj_vol_cont_adj_DB_wave14_female'],
						moments_male['corr_nondef_share_adj_vol_cont_adj_DC_wave10_male'],
						moments_male['corr_nondef_share_adj_vol_cont_adj_DB_wave10_male'],
						moments_female['corr_nondef_share_adj_vol_cont_adj_DC_wave10_female'],
						moments_female['corr_nondef_share_adj_vol_cont_adj_DB_wave10_female'],
						moments_male['corr_account_type_vol_cont_adj_wave14_male'],
						moments_female['corr_account_type_vol_cont_adj_wave14_female'],
						moments_male['corr_account_type_vol_cont_adj_wave10_male'],
						moments_female['corr_account_type_vol_cont_adj_wave10_female'],
						moments_female['corr_account_type_nondef_share_adj_wave14_female'],
						moments_female['corr_account_type_nondef_share_adj_wave10_female'],
						moments_male['mean_risky_risk_share_c_wave14_male'],
						moments_female['mean_risky_risk_share_c_wave14_female'],
						moments_male['mean_risky_risk_share_c_wave10_male'], 
						moments_female['mean_risky_risk_share_c_wave10_female'],
						moments_male['mean_nondef_share_adj_wave14_male'],
						moments_female['mean_nondef_share_adj_wave14_female'],
						moments_male['mean_nondef_share_adj_wave10_male'], 
						moments_female['mean_nondef_share_adj_wave10_female']], axis =1)
						#moments_male['mean_nondef_share_adj_wave14_male'],
						#moments_female['mean_nondef_share_adj_wave14_female'],
						#moments_male['mean_nondef_share_adj_wave10_male'], 
						#moments_female['mean_nondef_share_adj_wave10_female']], axis =1)
						#moments_male['mean_mortgagebal_wave14_male'],
						#moments_female['mean_mortgagebal_wave14_female'],
						#moments_male['mean_mortgagebal_wave10_male'], 
						#moments_female['mean_mortgagebal_wave10_female']],
						# )

	return moments_sorted 


def sortmoments_plot(moments_male, moments_female):

	empty = pd.DataFrame(np.zeros(9))

	moments_sorted = pd.concat(
						[
						moments_male['sd_consumption_wave14_male'],
						moments_female['sd_consumption_wave14_female'],
						moments_male['sd_consumption_wave10_male'],
						moments_female['sd_consumption_wave10_female'],
						moments_male['mean_account_type_wave14_male'],
						moments_female['mean_account_type_wave14_female'],
						moments_male['mean_account_type_wave10_male'],
						moments_female['mean_account_type_wave10_female'],
						moments_male['corr_super_balance_risky_share_adj_DC_wave14_male'],
						moments_male['corr_super_balance_risky_share_adj_DB_wave14_male'],
						moments_female['corr_super_balance_risky_share_adj_DC_wave14_female'],
						moments_female['corr_super_balance_risky_share_adj_DB_wave14_female'],
						moments_male['corr_super_balance_risky_share_adj_DC_wave10_male'],
						moments_male['corr_super_balance_risky_share_adj_DB_wave10_male'],
						moments_female['corr_super_balance_risky_share_adj_DC_wave10_female'],
						moments_female['corr_super_balance_risky_share_adj_DB_wave10_female'],
						moments_male['corr_super_balance_risk_share_DC_wave14_male'],
						moments_male['corr_super_balance_risk_share_DB_wave14_male'],
						moments_female['corr_super_balance_risk_share_DC_wave14_female'],
						moments_female['corr_super_balance_risk_share_DB_wave14_female'],
						moments_male['corr_super_balance_risk_share_DC_wave10_male'],
						moments_male['corr_super_balance_risk_share_DB_wave10_male'],
						moments_female['corr_super_balance_risk_share_DC_wave10_female'],
						moments_female['corr_super_balance_risk_share_DB_wave10_female'],
						moments_male['corr_super_balance_vol_cont_adj_DC_wave14_male'],
						moments_male['corr_super_balance_vol_cont_adj_DB_wave14_male'],
						moments_female['corr_super_balance_vol_cont_adj_DC_wave14_female'],
						moments_female['corr_super_balance_vol_cont_adj_DB_wave14_female'],
						moments_male['corr_super_balance_vol_cont_adj_DC_wave10_male'],
						moments_male['corr_super_balance_vol_cont_adj_DB_wave10_male'],
						moments_female['corr_super_balance_vol_cont_adj_DC_wave10_female'],
						moments_female['corr_super_balance_vol_cont_adj_DB_wave10_female'],
						moments_male['corr_vol_total_super_balance_DC_wave14_male'],
						moments_male['corr_vol_total_super_balance_DB_wave14_male'],
						moments_female['corr_vol_total_super_balance_DC_wave14_female'],
						moments_female['corr_vol_total_super_balance_DB_wave14_female'],
						moments_male['corr_vol_total_super_balance_DC_wave10_male'],
						moments_male['corr_vol_total_super_balance_DB_wave10_male'],
						moments_female['corr_vol_total_super_balance_DC_wave10_female'],
						moments_female['corr_vol_total_super_balance_DB_wave10_female'],
						moments_male['corr_consumption_wealth_real_wave14_male'],
						moments_female['corr_consumption_wealth_real_wave14_female'],
						moments_male['corr_consumption_wealth_real_wave10_male'],
						moments_female['corr_consumption_wealth_real_wave10_female'],
						moments_male['wealth_real_autocorr_male'],
						moments_female['wealth_real_autocorr_female'],
						moments_male['wealth_fin_autocorr_male'],
						moments_female['wealth_fin_autocorr_female'],
						moments_male['consumption_autocorr_male'],
						moments_female['consumption_autocorr_female'],
						moments_male['sd_risk_shareDC_wave14_male'],
						moments_male['sd_risk_shareDB_wave14_male'],
						moments_female['sd_risk_shareDC_wave14_female'],
						moments_female['sd_risk_shareDB_wave14_female'],
						moments_male['sd_risk_shareDC_wave10_male'],
						moments_male['sd_risk_shareDB_wave10_male'],
						moments_female['sd_risk_shareDC_wave10_female'],
						moments_female['sd_risk_shareDB_wave10_female'],
						moments_male['sd_vol_totalDC_wave14_male'],
						moments_male['sd_vol_totalDB_wave14_male'],
						moments_female['sd_vol_totalDC_wave14_female'],
						moments_female['sd_vol_totalDB_wave14_female'],
						moments_male['sd_vol_totalDC_wave10_male'],
						moments_male['sd_vol_totalDB_wave10_male'],
						moments_female['sd_vol_totalDC_wave10_female'],
						moments_female['sd_vol_totalDB_wave10_female'],
						moments_male['sd_super_balanceDC_wave14_male'],
						moments_male['sd_super_balanceDB_wave14_male'],
						moments_female['sd_super_balanceDC_wave14_female'],
						moments_female['sd_super_balanceDB_wave14_female'],
						moments_male['sd_super_balanceDC_wave10_male'],
						moments_male['sd_super_balanceDB_wave10_male'],
						moments_female['sd_super_balanceDC_wave10_female'],
						moments_female['sd_super_balanceDB_wave10_female'],
						moments_male['sd_wealth_real_wave14_male'],
						moments_female['sd_wealth_real_wave14_female'],
						moments_male['sd_wealth_real_wave10_male'],
						moments_female['sd_wealth_real_wave10_female'],
						moments_male['sd_wealth_fin_wave14_male'],
						moments_female['sd_wealth_fin_wave14_female'],
						moments_male['sd_wealth_fin_wave10_male'],
						moments_female['sd_wealth_fin_wave10_female'],
						moments_male['mean_vol_cont_c_wave14_male'],
						moments_female['mean_vol_cont_c_wave14_female'],
						moments_male['mean_vol_cont_c_wave10_male'],
						moments_female['mean_vol_cont_c_wave10_female'],
						moments_male['mean_risk_share_wave14_male'],
						moments_female['mean_risk_share_wave14_female'],
						moments_male['mean_risk_share_wave10_male'],
						moments_female['mean_risk_share_wave10_female'],
						moments_male['mean_vol_cont_adj_wave14_male'],
						moments_female['mean_vol_cont_adj_wave14_female'],
						moments_male['mean_vol_cont_adj_wave10_male'],
						moments_female['mean_vol_cont_adj_wave10_female'],
						moments_male['mean_vol_total_wave14_male'],
						moments_female['mean_vol_total_wave14_female'],
						moments_male['mean_vol_total_wave10_male'],
						moments_female['mean_vol_total_wave10_female'],
						moments_male['mean_wealth_real_wave14_male'],
						moments_female['mean_wealth_real_wave14_female'],
						moments_male['mean_wealth_real_wave10_male'],
						moments_female['mean_wealth_real_wave10_female'],
						moments_male['mean_wealth_fin_wave14_male'],
						moments_female['mean_wealth_fin_wave14_female'],
						moments_male['mean_wealth_fin_wave10_male'],
						moments_female['mean_wealth_fin_wave10_female'],
						moments_male['mean_super_balance_wave14_male'],
						moments_female['mean_super_balance_wave14_female'],
						moments_male['mean_super_balance_wave10_male'],
						moments_female['mean_super_balance_wave10_female'],
						moments_male['mean_consumption_wave14_male'],
						moments_female['mean_consumption_wave14_female'],
						moments_male['mean_consumption_wave10_male'], 
						moments_female['mean_consumption_wave10_female'],
						moments_male['corr_nondef_share_adj_vol_cont_adj_DC_wave14_male'],
						moments_male['corr_nondef_share_adj_vol_cont_adj_DB_wave14_male'],
						moments_female['corr_nondef_share_adj_vol_cont_adj_DC_wave14_female'],
						moments_female['corr_nondef_share_adj_vol_cont_adj_DB_wave14_female'],
						moments_male['corr_nondef_share_adj_vol_cont_adj_DC_wave10_male'],
						moments_male['corr_nondef_share_adj_vol_cont_adj_DB_wave10_male'],
						moments_female['corr_nondef_share_adj_vol_cont_adj_DC_wave10_female'],
						moments_female['corr_nondef_share_adj_vol_cont_adj_DB_wave10_female'],
						moments_male['corr_account_type_vol_cont_adj_wave14_male'],
						moments_female['corr_account_type_vol_cont_adj_wave14_female'],
						moments_male['corr_account_type_vol_cont_adj_wave10_male'],
						moments_female['corr_account_type_vol_cont_adj_wave10_female'],
						moments_female['corr_account_type_nondef_share_adj_wave14_female'],
						moments_female['corr_account_type_nondef_share_adj_wave10_female'],
						moments_male['mean_risky_risk_share_c_wave14_male'],
						moments_female['mean_risky_risk_share_c_wave14_female'],
						moments_male['mean_risky_risk_share_c_wave10_male'], 
						moments_female['mean_risky_risk_share_c_wave10_female'],
						moments_male['mean_nondef_share_adj_wave14_male'],
						moments_female['mean_nondef_share_adj_wave14_female'],
						moments_male['mean_nondef_share_adj_wave10_male'], 
						moments_female['mean_nondef_share_adj_wave10_female'],
						moments_male['mean_mortgagebal_wave14_male'],
						moments_female['mean_mortgagebal_wave14_female'],
						moments_male['mean_mortgagebal_wave10_male'], 
						moments_female['mean_mortgagebal_wave10_female'],
						1- moments_male['mean_renter_wave14_male'],
						1- moments_female['mean_renter_wave14_female'],
						1- moments_male['mean_renter_wave10_male'], 
						1- moments_female['mean_renter_wave10_female'],
						moments_male['mean_house_adj_wave14_male'],
						moments_female['mean_house_adj_wave14_female'],
						moments_male['mean_house_adj_wave10_male'], 
						moments_female['mean_house_adj_wave10_female'],
						(moments_male['mean_Adjustment_cost_v_wave14_male']),
						moments_female['mean_Adjustment_cost_v_wave14_female'],
						(moments_male['mean_Adjustment_cost_v_wave10_male']), 
						moments_female['mean_Adjustment_cost_v_wave10_female'],
						(moments_male['mean_Adjustment_cost_pi_wave14_male']),
						moments_female['mean_Adjustment_cost_pi_wave14_female'],
						moments_male['mean_Adjustment_cost_pi_wave10_male'], 
						moments_female['mean_Adjustment_cost_pi_wave10_female']],axis =1)
						#(moments_male['mean_Adjustment_cost_plan_wave14_male']),
						#(moments_female['mean_Adjustment_cost_plan_wave14_female']),
						#(moments_male['mean_Adjustment_cost_plan_wave10_male']), 
						#moments_female['mean_Adjustment_cost_plan_wave10_female']] 
						 
	return moments_sorted 

if __name__ == '__main__':
	
	import seaborn as sns
	import yaml
	import dill as pickle
	import glob 
	import copy
	import os
	import time
	import gc
	import matplotlib.pyplot as plt

	# Housing model modules
	#sys.path.append("..")
	import lifecycle_model 
	import numpy as np

	simlist_f = ['female_4']
	simlist_m = ['male_16']
	sim_id = "male_16"

	for sim_id_m, sim_id_f  in zip(simlist_m,simlist_f):
		if not os.path.isdir(sim_id):
			os.makedirs(sim_id)

		plt.close()

		sns.set(font_scale =1.5,style='ticks',rc={"lines.linewidth": 0.7, \
			"axes.grid.axis":"both","axes.linewidth":2,"axes.labelsize":18})
		plt.rcParams["figure.figsize"] = (20,10)
		sns.set_style('white')


		def create_plot(labelsx,lab_str,df,col1,col2, source, marker,color, ylim, ylabel):
			#df[col1] = df[col1].map(lambda x :inc_y0(x))
			#sns.set()
			 #list(set(df[source]))
			line_names = ['wave14_data', 'wave10_data', 'wave14_sim', 'wave10_sim']
			linestyles = ["-","-","-","-"]
			col_dict = {'wave14_data': 'black', 'wave10_data':'black', 'wave14_sim':'gray', 'wave10_sim':'gray'}

			normalise_list = ['sd_vol_totalDC','sd_vol_totalDB','sd_super_balanceDC','sd_super_balanceDB','mean_wealth_real','mean_wealth_fin',\
					'mean_super_balance',\
					'mean_vol_total',\
					'mean_vol_cont_c',\
					'mean_consumption','sd_consumption', \
					'sd_wealth_real', 'sd_wealth_fin']


			df_male = df.iloc[:,[0, 1,3,5,7]]
			df_female = df.iloc[:,[0, 2,4,6,8]] 


			df_male.columns = df_male.columns.str.replace("_male", "")
			df_female.columns = df_female.columns.str.replace("_female", "")



			df_male = df_male.melt('Age_wave10', var_name= 'source', value_name = key)
			df_male['source'] = df_male['source'].str.replace(key+"_", "")
			df_female = df_female.melt('Age_wave10', var_name= 'source', value_name = key)
			df_female['source'] = df_female['source'].str.replace(key+"_", "")
			

			markers = ['x', 'o', 'x', 'o']

			if col2 in normalise_list:
				df_male[col2] = df_male[col2].div(1000)
				df_female[col2] = df_female[col2].div(1000)

			figure, axes = plt.subplots(nrows=1, ncols=2)

			for name, marker, linestyle in zip(line_names, markers, linestyles):
						data_male = df_male.loc[df_male[source]==name]
						data_female = df_female.loc[df_female[source]==name]
						xs = list(labelsx)
						print(xs)
						if col2 != 'mean_Adjustment_cost_v' and col2 != 'mean_Adjustment_cost_pi':
							ys = list(data_male[col2])[0:18]
							p = axes[0].plot(xs, ys, marker=marker, color=col_dict[name], linestyle=linestyle,
									 label=name, linewidth=2)
							ys = list(data_female[col2])[0:18]
							p = axes[1].plot(xs, ys, marker=marker, color=col_dict[name], linestyle=linestyle,
									label=name, linewidth=2)
						elif name == 'wave14_sim' or name == 'wave10_sim':
							ys = list(data_male[col2])[0:18]
							p = axes[0].plot(xs, ys, marker=marker, color=col_dict[name], linestyle=linestyle,
									 label=name, linewidth=2)
							ys = list(data_female[col2])[0:18]
							p = axes[1].plot(xs, ys, marker=marker, color=col_dict[name], linestyle=linestyle,
									label=name, linewidth=2)
						else:
							pass 

			
			if isinstance(ylabel, str):
						ylabel = ylabel
			
			axes[0].set_title('Males')
			axes[0].set_xlabel('Age cohort')
			#axes[0].set_ylabel(ylabel)
			
			axes[0].spines['top'].set_visible(False)
			axes[0].spines['right'].set_visible(False)
			axes[0].legend(loc='bottom left', ncol=2)
			#axes[0].update_datalim(np.c_[xs,[0]*len(xs)], updatey=False)
			axes[0].set_ylim(ylim)
			#axes[0].autoscale()
			axes[0].set_xlim(min(xs), max(xs))
			axes[0].set_xticklabels(lab_str)
			#plt.show()
			axes[1].set_title('Females')
			axes[1].set_xlabel('Age cohort')
		   #axes[1].set_ylabel(ylabel)
			
			axes[1].spines['top'].set_visible(False)
			axes[1].spines['right'].set_visible(False)
			axes[1].legend(loc='bottom left', ncol=2)
			#axes[1].update_datalim(np.c_[xs,[0]*len(xs)], updatey=False)
			axes[1].set_ylim(ylim)
		   # axes[1].autoscale()
			axes[1].set_xlim(min(xs), max(xs))
			axes[1].set_xticklabels(lab_str)
			
			
			#plt.tight_layout()
			#figure.size(10,10)
			#figure.subplots_adjust(right=1.25)
			figure.savefig("plots/{}/{}.png".format(sim_id,col2), transparent=True)
		   

		# Read settings
		# Make sure the og class has the parameters consistent with the policy that is being loaded!
		with open("settings/settings.yml", "r") as stream:
			eggbasket_config = yaml.safe_load(stream)
	   
		# Get best male moments
		model_name = 'baseline_male'

		top_id = pickle.load(open("/scratch/pv33/ls_model_temp2/baseline_male/topid.smms".format(model_name),"rb"))
		#top_id = '9GLB6E_20210501-181303_baseline_male'
		params = pickle.load(open("/scratch/pv33/ls_model_temp2/baseline_male/{}_acc_0/params.smms".format(top_id),"rb"))
		param_dict = eggbasket_config['male']
		param_dict['parameters'] = params

		og = lifecycle_model.LifeCycleModel(param_dict,
						np.array([0]), param_id = top_id, mod_name = model_name)
		settings_folder = 'settings/'
		generate_TSDF = genprofiles_operator(og)

		# generate random numbers for the simulation
		TSN = 250
		U = np.random.rand(6,100,TSN,100) 

		TSALL_10_df, TSALL_14_df = gen_panel_ts('male',og,U, TSN)
		moments = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df))
		moments_male =  copy.copy(moments).add_suffix('_male')
		moments_female = copy.copy(moments).add_suffix('_female') 
		
		# Get best female moments
		#model_name = 'baseline_female'
		#top_id = pickle.load(open("/scratch/pv33/ls_model_temp2/{}/single_ID_latest.smms".format(model_name),"rb"))
		#params = pickle.load(open("/scratch/pv33/ls_model_temp2/{}/{}_acc_0/params.smms".format(model_name, top_id),"rb"))
		#param_dict = eggbasket_config['female']
		#param_dict['parameters'] = params

		#og = lifecycle_model.LifeCycleModel(param_dict,
		 #               np.array([0]), param_id = top_id, mod_name = model_name)
		#settings_folder = 'settings/'

		# generate random numbers for the simulation
		#TSN = 350
		#U = np.random.rand(6,100,TSN,100) 

		#TSALL_10_df, TSALL_14_df = gen_panel_ts('females',og,U, TSN,False, .19)
		#moments_female = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_female') 
		  
		
		os.chdir('/home/141/as3442/Eggsandbasket2.0/eggsandbaskets')  
		moments_male.to_csv("moments_male.csv") 
		moments_female.to_csv("moments_female.csv") 
		#moments_male = pd.read_csv("moments_male.csv") 
		#moments_female = pd.read_csv("moments_female.csv") 

		moments_sorted  = sortmoments_plot(moments_male, moments_female)
		moments_sorted = pd.concat([moments_male["Age_wave10_male"].reset_index().iloc[:,1], moments_sorted], axis =1)  
		moments_sorted = moments_sorted.rename(columns = {'Age_wave10_male':'Age_wave10'})
		moments_sorted.to_csv("plots/moments_sorted.csv".format(model_name))  
		moments_data = pd.read_csv('{}moments_data_plot.csv'\
					.format(settings_folder))
		moments_data.columns = moments_sorted.columns
		age = np.arange(18, 65) # PROBABLY SHOULD GET RID OF AGE MAGIC NUMBERS HERE 

		moments_sorted[moments_data.isnull()] = float('nan')
		plot_keys_vars  = ['mean_account_type',
							'corr_super_balance_risky_share_adj_DC',
							'corr_super_balance_risky_share_adj_DB',
							'corr_super_balance_risk_share_DC',
							'corr_super_balance_risk_share_DB',
							'corr_super_balance_vol_cont_adj_DC',
							'corr_super_balance_vol_cont_adj_DB',
							'corr_vol_total_super_balance_DC',
							'corr_vol_total_super_balance_DB',
							'corr_consumption_wealth_real',
							'sd_risk_shareDC',
							'sd_risk_shareDB',
							'sd_vol_totalDC',
							'sd_vol_totalDB',
							'sd_super_balanceDC',
							'sd_super_balanceDB',
							'sd_wealth_real',
							'sd_wealth_fin',
							'mean_risk_share',
							'mean_renter',
							'mean_house_adj',
							'mean_vol_cont_adj',
							'mean_vol_total',
							'mean_vol_cont_c',
							'mean_wealth_real',
							'mean_wealth_fin',
							'mean_super_balance',
							'mean_consumption', 
							'sd_consumption', 
							'mean_Adjustment_cost_v', 
							'mean_Adjustment_cost_pi']


		plot_autocors = ['wealth_real_autocorr',
							'wealth_fin_autocorr',
							'consumption_autocorr']

		# variables with y axis 0:1000
		# real wealth, fin wealth super balance
		axis_style_list_ylim = {
							'mean_Adjustment_cost_v':(0,300),
							'mean_Adjustment_cost_pi':(0,20),
							'mean_account_type':(0,1),
							'sd_super_balanceDC':(0,1500),
							'sd_super_balanceDB':(0,1500),
							'sd_wealth_real':(0,1500),
							'sd_wealth_fin':(0,1500),
							'mean_wealth_real':(0,1500),
							'mean_wealth_fin':(0,1500),
							'mean_super_balance':(0,1500),
		# varibales with y 1- 150 (consumption)
							'mean_consumption':(0,150),
							'sd_consumption':(0,150),
		# varibales with y axis 0 -1(risky share and vol cont)
							'mean_vol_cont_adj':(0,1),
							'mean_risk_share':(0,1),
							'mean_renter':(0,1),
							'mean_house_adj':(0,1),
							'sd_risk_shareDC':(0,1),
							'sd_risk_shareDB':(0,1),
							'sd_vol_totalDC':(0,15),
							'sd_vol_totalDB':(0,15),
							'mean_vol_total': (0,15),
							'mean_vol_cont_c': (0,15),
		# varibales with y axis -.5 to 1(correlations)
							'corr_super_balance_risky_share_adj_DC':(-.8,1),
							'corr_super_balance_risky_share_adj_DB':(-.8,1),
							'corr_super_balance_risk_share_DC':(-.8,1),
							'corr_super_balance_risk_share_DB':(-.8,1),
							'corr_super_balance_vol_cont_adj_DC':(-.8,1),
							'corr_super_balance_vol_cont_adj_DB':(-.8,1),
							'corr_vol_total_super_balance_DC':(-.8,1),
							'corr_vol_total_super_balance_DB':(-.8,1),
							'corr_consumption_wealth_real':(-.8,1)}

		axis_label_list = { 'mean_Adjustment_cost_v': 'Utility loss from adj vol. cont',
							'mean_Adjustment_cost_pi': 'Utility loss from adj risk',
							'mean_account_type':'Share of DC',
							'sd_super_balanceDC':'SD: UniSuper balance (DC)',
							'sd_super_balanceDB':'SD: UniSuper balance (DB)',
							'mean_wealth_real': 'Housing asset',
							'mean_wealth_fin':'Asset',
							'mean_super_balance':'UniSuper balance (DB+DC)',
							'sd_wealth_real':'SD: Asset',
							'sd_wealth_fin' :'SD: Housing asset',
		# varibales with y 1- 150 (consumption)
							'mean_consumption':'Consumption',
							'sd_consumption': 'SD: Consumption',
		# varibales with y axis 0 -1(risky share and vol cont)
							'mean_vol_cont_adj':'Share of positive voluntary contribution',
							'mean_risk_share': 'Share of Unisuper balance in risk assets',
							'mean_renter': 'Share of homeowners',
							'mean_house_adj': 'Share adjusting housing',
							'sd_risk_shareDC': 'SD: Share of Unisuper balance in risk assets among DCer',
							'sd_risk_shareDB': 'SD: Share of Unisuper balance in risk assets among DBer',
							'sd_vol_totalDC':'SD: total voluntary contribution among DCer',
							'sd_vol_totalDB':'SD: total voluntary contribution among DBer',
							'mean_vol_total': 'Total voluntary contribution',
							'mean_vol_cont_c':'Total voluntary contribution (among contributors)',
		# varibales with y axis -.5 to 1(correlations)
							'corr_super_balance_risky_share_adj_DC':'CORR: UniSuper balane and non−default inv among DCer',
							'corr_super_balance_risky_share_adj_DB':'CORR: UniSuper balane and non−default inv among DBer',
							'corr_super_balance_risk_share_DC':'CORR: UniSuper balane and risky share among DCer',
							'corr_super_balance_risk_share_DB':'CORR: UniSuper balane and risky share among DCer',
							'corr_super_balance_vol_cont_adj_DC':'CORR: UniSuper balane and +vc among DCer',
							'corr_super_balance_vol_cont_adj_DB':'CORR: UniSuper balane and +vc among DBer',
							'corr_vol_total_super_balance_DC':'CORR: UniSuper balane and vc among DCer',
							'corr_vol_total_super_balance_DB':'CORR: UniSuper balane and vc among DCer',
							'corr_consumption_wealth_real':'CORR: Consumption and real wealth'}


		#excludes   = set(['account_type_all'])
		cohort_labels = pd.DataFrame(list(["<25",
						"25-29",
						"30-34",
						"35-39",
						"40-44",
						"45-49",
						"50-54",
						"55-59",
						"60+",
						]))
		cohort_labels.columns= ['Age_wave10'] 

		labels = list(["1",
						"2",
						"3",
						"4",
						"5",
						"6",
						"7",
						"8",
						"9",
						])
		labels_str =  list(["<25",
						"25-29",
						"30-34",
						"35-39",
						"40-44",
						"45-49",
						"50-54",
						"55-59",
						"60+",
						])

		for  key in plot_keys_vars:

			#Assets 
			var_data = moments_data[[key+'_wave10_male',key+'_wave10_female',key+'_wave14_male',key+'_wave14_female']]
			var_data = var_data.add_suffix('_data')
			var_data.iloc[8,2] = float('nan')
			var_data.iloc[8,3] = float('nan')  

			var_sim = moments_sorted[[key+'_wave10_male',key+'_wave10_female',key+'_wave14_male',key+'_wave14_female']]
			var_sim = var_sim.add_suffix('_sim')
			var_sim.iloc[8,2] = float('nan')
			var_sim.iloc[8,3] = float('nan')  


			var_grouped = pd.concat([cohort_labels, var_data, var_sim], axis =1)
	 
			ylim = axis_style_list_ylim[key]

			create_plot(labels,labels_str,var_grouped, 'Age_wave10', key, 'source', marker='s',color='darkblue', ylim = ylim, ylabel = axis_label_list[key])

		# plot adj cost

		figure, axes = plt.subplots(nrows=1, ncols=2)

		xs = list(labels)[0:9]

		axes[0].plot(xs,1e-3*moments_sorted['mean_Adjustment_cost_v_wave10_male'][0:9]*.5 + 1e-3*moments_sorted['mean_Adjustment_cost_v_wave14_male'][0:9]*.5, marker='', color='black', linestyle='dashed',
									 label='Voluntary contribution', linewidth=2)
		axes[0].plot(xs,1e-3*moments_sorted['mean_Adjustment_cost_pi_wave10_male'][0:9]*.5 + 1e-3*moments_sorted['mean_Adjustment_cost_pi_wave14_male'][0:9]*.5, marker='', color='black', linestyle='dotted',
									 label='Investment choice', linewidth=2)
		axes[0].plot(xs,1e-3*moments_sorted['mean_Adjustment_cost_plan_wave10_male'][0:9]*.5 + 1e-3*moments_sorted['mean_Adjustment_cost_plan_wave14_male'][0:9]*.5, marker='', color='black', linestyle='-',
									 label='Pension type', linewidth=2)

		axes[1].plot(xs,1e-3*moments_sorted['mean_Adjustment_cost_v_wave10_female'][0:9]*.5 + 1e-3*moments_sorted['mean_Adjustment_cost_v_wave14_female'][0:9]*.5, marker='', color='black', linestyle='dashed',
									 label='Voluntary contribution', linewidth=2)
		axes[1].plot(xs,1e-3*moments_sorted['mean_Adjustment_cost_pi_wave10_female'][0:9]*.5 + 1e-3*moments_sorted['mean_Adjustment_cost_pi_wave14_female'][0:9]*.5, marker='', color='black', linestyle='dotted',
									 label='Investment choice', linewidth=2)
		axes[1].plot(xs,1e-3*moments_sorted['mean_Adjustment_cost_plan_wave10_female'][0:9]*.5 + 1e-3*moments_sorted['mean_Adjustment_cost_plan_wave14_female'][0:9]*.5, marker='', color='black', linestyle='-',
									 label='Pension type', linewidth=2)

		axes[0].set_title('Males')
		axes[0].set_xlabel('Age cohort')
		#axes[0].set_ylabel(ylabel)

		axes[0].spines['top'].set_visible(False)
		axes[0].spines['right'].set_visible(False)
		axes[0].legend(loc='bottom left', ncol=1)
		#axes[0].update_datalim(np.c_[xs,[0]*len(xs)], updatey=False)
		
		axes[0].autoscale()
		axes[0].set_xlim(min(xs), max(xs))
		axes[0].set_xticklabels(labels_str[0:9])
		axes[0].set_ylim(0,140)
		#plt.show()
		axes[1].set_title('Females')
		axes[1].set_xlabel('Age cohort')
		#axes[1].set_ylabel(ylabel)

		axes[1].spines['top'].set_visible(False)
		axes[1].spines['right'].set_visible(False)
		axes[1].legend(loc='bottom left', ncol=1)
		#axes[1].update_datalim(np.c_[xs,[0]*len(xs)], updatey=False)
		
		axes[1].autoscale()
		axes[1].set_xlim(min(xs), max(xs))
		axes[1].set_xticklabels(labels_str)
		axes[1].set_ylim(0,140)


		#plt.tight_layout()
		#figure.size(10,10)
		#figure.subplots_adjust(right=1.25)
		figure.savefig("plots/male_16/adj_costs.png", transparent=True)
