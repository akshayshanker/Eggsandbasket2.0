
"""
This module contains the functions to generate time series of agents
and moments from a solved HousingModel 
 
Functions: genprofiles
			Generates the a time series of N agents from the HousingModel class 
		   
		   housingmodel_function_factory
			Generates moments from time-series 

"""

import numpy as np
from quantecon import tauchen
import matplotlib.pyplot as plt
from itertools import product
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
import pandas as pd
import copy
import dill as pickle 
from numba import njit
from itertools import permutations
from interpolation.splines import extrap_options as xto
import gc

from eggsandbaskets.util.helper_funcs import *

def genprofiles_operator(og,
				norm = 1E5):

	

	"""For a solved housing model, returns a time-series of profiles 

	Parameters
	----------
	og: class
		Housing nmodel  

	N: int
		Number of agents to simulate 

	Returns
	-------
	TSALL
		Dictionary containing time-series
	"""
	
	# unpack and define functions to be used in time-series generator 
	# declare functions

	u, uc, uh,uc_inv, uh_inv        = og.u, og.uc, og.uh, og.uc_inv, og.uh_inv
	b, b_prime                      = og.b, og.b_prime 
	y, DB_benefit                   = og.y, og.DB_benefit
	adj_p, adj_v,adj_pi             = og.adj_p, og.adj_v, og.adj_pi
	sigma_plan 						= og.sigma_plan


	beta,   delta_housing, alpha    = og.beta, og.delta_housing, og.alpha
	tau_housing, def_pi             = og.tau_housing, og.def_pi     
	v_S,    v_E                     = og.v_S,og.v_E 
	r,s, r_H                        = og.r, og.s, og.r_H     
	alpha_housing                   = og.alpha_housing

	Q_shocks_r, Q_shocks_P          = og.Q_shocks_r, og.Q_shocks_P
	Q_DC_shocks, Q_DC_P             = og.Q_DC_shocks, og.Q_DC_P 
	EBA_P                           = og.EBA_P
	X_QH_R,X_QH_W                   = og.X_QH_R, og.X_QH_W 

	beta_hat, P_beta, beta_stat     = og.beta_hat, og.P_beta, og.beta_stat 
	alpha_hat, P_alpha, alpha_stat  = og.alpha_hat, og.P_alpha, og.alpha_stat 
	beta_bar, alpha_bar             = og.beta_bar, og.alpha_bar

	Q = og.Q

	sigma_DC_V, sigma_DB_V          = og.sigma_DC_V, og.sigma_DB_V  
	sigma_DC_pi, sigma_DB_pi        = og.sigma_DC_pi, og.sigma_DB_pi

	E, P_E, P_stat                  = og.E, og.P_E, og.P_stat
	A_DC                            = og.A_DC
	H                               = og.H  
	A                               = og.A
	A_min                           = og.A_min
	H_min, H_max, grid_size_H       = og.H_min, og.H_max, og.grid_size_H
	A_max_R                         = og.A_max_R
	X_H_R                           = og.X_H_R_ind 
	W_R                             = og.W_R
	W_W                             = og.W_W


	V, Pi                           = og.V, og.Pi
	X_r, P_r                        = og.X_r, og.P_r
	X_disc, X_disc_vals, X_disc_exog= og.X_disc, og.X_disc_vals,\
										 og.X_disc_exog
	X_d_exogNA_cont_vals            = og.X_d_exogNA_cont_vals
	X_d_ex_cont                     = og.X_d_ex_cont
	X_all_bar_vals                  = og.X_all_bar_vals
	X_W_NA_grid_ind                 = og.X_W_NA_grid_ind
	g_size_X_W_all                  = og.g_size_X_W_all

	X_DCQ_W                         = og.X_DCQ_W

	X_cont_W, X_cont_R              = og.X_cont_W, og.X_cont_R 
	X_W_contgp, X_R_contgp          = og.X_W_contgp, og.X_R_contgp 
	X_all_ind_W                     = og.X_all_ind_W
	X_d_ex_cont                     = og.X_d_ex_cont
	E_index, V_index, Pi_index      = og.E_index, og.V_index, og.Pi_index                                  

	grid_size_A, grid_size_DC, grid_size_H, grid_size_Q = og.grid_size_A,\
									  og.grid_size_DC,\
									  og.grid_size_H,\
									  og.grid_size_Q

	grid_size_W, grid_size_alpha, grid_size_beta = int(og.grid_size_W),\
										int(og.grid_size_alpha),\
										int(og.grid_size_beta)
	
	T, tzero, R                     = og.T, int(og.tzero), og.R
	A_max_W, H_max                  = og.A_max_W, og.H_max
	A_R, H_Q, A_Q_R                 = og.A_R, og.H_Q, og.A_Q_R
	X_Q_R_ind                       = og.X_Q_R_ind
	X_R_cont_ind                    = og.X_R_cont_ind    
	X_H_R_ind                       = og.X_H_R_ind      

	DC_max                          = og.DC_max
	Q_max                           = og.Q_max
	M_shock_index                   = og.M_shock_index

	grid_AW                         = og.grid_AW

	A_R_H                           = og.A_R_H

	labour_mc						= og.labour_mc
	beta_mc 						= og.beta_mc
	alpha_mc 						= og.alpha_mc

	r_h 							= og.r_h
	r_l 							= og.r_l

	# unpack policies

	@njit 
	def gen_VPi(points, Age, account_ind,\
				E_ind, alpha_ind,beta_ind,\
				pi_ushock, v_ushock,Xi_cov,Xi_copi):

		""" Genrates voluntary contribution 
			risk prob"""

		xi_V_vals = eval_linear(X_cont_W, Xi_cov[Age][account_ind,E_ind,alpha_ind,beta_ind,:],\
								 points) 

		xi_V_vals        = xi_V_vals - max(xi_V_vals)  

		prob_v           = np.exp(xi_V_vals)/np.sum(np.exp(xi_V_vals))

		#pick a random draw for the voluntary contribution (index in the vol.cont grid)
		V_ind 			= np.arange(len(V))[np.searchsorted(np.cumsum(prob_v), v_ushock)]

		v 	  			= V[V_ind]

		#pull our probabiltiy of risk share matrix for this cont state 
		xi_Pi_vals 		= eval_linear(X_cont_W, Xi_copi[Age][account_ind,E_ind,alpha_ind,beta_ind,V_ind,:],\
										points) 

		xi_Pi_vals        = xi_Pi_vals - max(xi_Pi_vals)  
		prob_Pi           = np.exp(xi_Pi_vals)/np.sum(np.exp(xi_Pi_vals))


		Pi_ind 				= np.arange(len(Pi))[np.searchsorted(np.cumsum(prob_Pi), pi_ushock)]

		pi 	   				= Pi[Pi_ind]

		return V_ind, v, Pi_ind, pi 


	@njit
	def seriesgen(age,wave_length, W, beta_hat_ts, alpha_hat_ts,V_ushock_ts, Pi_ushock_ts,DBshock,\
					a_noadj, etas_noadj, H_adj, a_adj, c_adj, Xi_cov,Xi_copi,policy_VF):
		""" Returns a time series of one agent to age 
			in baseline year and age in baseline year + wave_len
		"""

		length 	= int(age + wave_length+2) 

		# generate sequences of shocks for this agent i 
		# remeber start year is tzero. 
		# we generate shocks such that TS_j[t] gives the value of
		# the shock at t

		TS_A,TS_A_1  	= 0,0
		TS_H, TS_H_1 	= 0,0
		TS_DC,TS_DC_1 	= 0,0 
		TS_C,TS_C_1  	= 0,0
		TS_V,TS_V_1  	= 0,0 
		TS_PI,TS_PI_1 	= 0,0
		TS_wage,TS_wage_1 = 0,0
		TS_hinv,TS_hinv_1 = 0,0
		adj_V,adj_V_1 	= 0,0
		adj_pi,adj_pi_1 	=0,0
		P_h	= np.zeros(length+1)


		# generate sequence of house prices

		P_h[tzero]  = 1/((1+r_H)**(age - tzero))

		for t in np.arange(tzero, len(P_h)-1):
			P_h[t+1] = (1+r_H)*P_h[t]  

		# initialize continuous points 

		TS_A = 1e-5
		TS_H= 1e-5
		TS_DC= 1e-5

		wave_data_14 = np.zeros(17)
		wave_data_10 = np.zeros(17)

		# generate time series of wage shocks, beta_hat and alpha_hat draws for this guy



		#DC_H= 	np.exp(r_h + lnrh_mc.simulate(ts_length = length)) # these need to be made deterministic 
		#DC_L = 	np.exp(r_l + lnrl_mc.simulate(ts_length = length)) # these need to be made deterministic 	


		# get the initial value of the series and draw from account type
		

		#sigma_plan =.03
		#disc_exog_ind = int(np.where((np.where(W[int(tzero)]== E)[0] \
		#						== X_disc_exog[:,0]) \
		#					&   (X_disc_exog[:,1] == np.where(alpha_hat \
		#						 == alpha_hat_ts[int(tzero)])[0]) \
		#					& 	(X_disc_exog[:,2] == np.where(beta_hat\
		#						 == beta_hat_ts[int(tzero)])[0]))[0])

		E_ind 		= int(W[tzero])
		beta_ind 	= int(beta_hat_ts[tzero])
		alpha_ind 	= int(alpha_hat_ts[tzero])


		V_DC 		=  eval_linear(X_cont_W, \
				 		policy_VF[1,E_ind,alpha_ind,beta_ind,:], \
				 		np.array([TS_A,TS_DC,TS_H, P_h[tzero]]))



		V_DB 		=  eval_linear(X_cont_W, \
				 		policy_VF[0,E_ind,alpha_ind,beta_ind,:], \
				 		np.array([TS_A,TS_DC,TS_H, P_h[tzero]]))

		V_DC_scaled = ((V_DC - adj_p(tzero))/sigma_plan) - max(((V_DC - adj_p(tzero))/sigma_plan), ((V_DB/sigma_plan)))
		V_DB_scaled = ((V_DB/sigma_plan)) 			 	- max(((V_DC - adj_p(tzero))/sigma_plan), ((V_DB/sigma_plan)))


		Prob_DC 	= np.exp(V_DC_scaled)/(
						np.exp(V_DB_scaled) \
					+   np.exp(V_DC_scaled ) )    


		account_ind 	= int(np.searchsorted(np.cumsum(np.array([1-Prob_DC, Prob_DC])), DBshock))

		for t in range(int(tzero), int(length)+1):
			if t<R:
				#get the index for the labour shock
				E_ind 		= int(W[t])
				beta_ind 	= int(beta_hat_ts[t])
				alpha_ind 	= int(alpha_hat_ts[t])

				E_val 		= E[int(W[t])]
				beta_val 	= beta_hat[int(beta_hat_ts[t])]
				alpha_val 	= alpha_hat[int(alpha_hat_ts[t])]

				#get continuous state index
				points = np.array([TS_A,TS_DC\
									,TS_H, P_h[t]])
				
				#pull out the probability of voluntary contribution probability matrix for this cont state

				pi_ushock = Pi_ushock_ts[t]
				v_ushock  = V_ushock_ts[t]
				args = (account_ind, E_ind, alpha_ind, beta_ind,pi_ushock, v_ushock,Xi_cov,Xi_copi)

				V_ind, v, Pi_ind, pi  = gen_VPi(points,t-tzero, *args)

				#calculate wage for agent 
				TS_wage	= y(t,E[E_ind])

				# total liquid wealth for non-adjuster 
				wealth_no_adj       = TS_A*(1+r)\
										+ (1-v -v_S -v_E)*TS_wage  

				# next period DC assets (before returns)
				# (recall domain of policy functions from def of eval_policy_W)
				# DC_prime is DC assets at the end of t, before returns into t+1

				DC_prime          = TS_DC + (v +(v_S +v_E)*account_ind)*TS_wage

				h                   = TS_H*(1-delta_housing)
				q 					=P_h[t]

				TS_DC_1 			= (1+(1-pi)*r_l\
										+ pi*r_h )*DC_prime  

				wealth_adj          = wealth_no_adj + P_h[t]*h

				eta_func            =  etas_noadj[t-tzero][account_ind,\
													E_ind,\
													alpha_ind,\
													beta_ind,\
													Pi_ind,:]
				
				eta                 = eval_linear(X_cont_W,eta_func,\
												  np.array([wealth_no_adj,\
												  DC_prime,h,P_h[t]]),\
												  xto.NEAREST)

				if np.abs(eta)<1:
					a_noadjust_func     =  a_noadj[t-tzero][account_ind,\
													E_ind,\
													alpha_ind,\
													beta_ind,\
													Pi_ind,:]
					h_prime  			= h
					TS_A_1  			= eval_linear(X_cont_W,a_noadjust_func,\
												np.array([wealth_no_adj,DC_prime,h,q]),\
												xto.NEAREST)
					TS_C        		= max(1e-10,wealth_no_adj\
												- TS_A_1)
					TS_H_1 		 	= h 


				else:
					a_prime_adj_func = a_adj[t-tzero][account_ind,\
													E_ind,\
													alpha_ind,\
													beta_ind,\
													Pi_ind,:]
					c_prime_adj_func = c_adj[t-tzero][account_ind,\
													E_ind,\
													alpha_ind,\
													beta_ind,\
													Pi_ind,:]

					H_adj_func = H_adj[t-tzero][account_ind,\
													E_ind,\
													alpha_ind,\
													beta_ind,\
													Pi_ind,:]


					TS_C           = max(eval_linear(X_QH_W,\
												c_prime_adj_func,\
												np.array([DC_prime,q,\
												wealth_adj]),xto.NEAREST), 1E-20)

					TS_H_1          = max(1e-20, eval_linear(X_QH_W,\
													H_adj_func,\
													np.array([DC_prime,q,\
													wealth_adj]), xto.NEAREST))

					TS_A_1          = max(1e-20, eval_linear(X_QH_W,\
													a_prime_adj_func,\
													np.array([DC_prime,q,\
													wealth_adj]), xto.NEAREST))
				TS_hinv = TS_H_1 -TS_H
				TS_PI   = pi
				TS_V	   = v

					# if t not terminal, iterate forward 
				if t== age:
					wave_data_10 			=  np.array([account_ind,age, TS_A*norm,\
											TS_H*norm, TS_DC*norm, TS_C*norm, \
											TS_wage*norm, TS_V, TS_V*TS_wage*norm, TS_PI, \
											int(TS_hinv>0), int(TS_PI>.7), int(TS_V>0), \
											alpha_hat_ts[age], beta_hat_ts[age],adj_V, adj_pi])


				if t==age + wave_length:
						# we denote the age at wave_14 by thier age at 10 so they go in 2010 bucket
					age_wave_10 			= age
					age14 					= age+ wave_length
					wave_data_14 			= np.array([account_ind,age_wave_10, TS_A*norm,\
												TS_H*norm, TS_DC*norm, TS_C*norm, \
												TS_wage*norm, TS_V,TS_V*TS_wage*norm, TS_PI, \
												int(TS_hinv>0), int(TS_PI>.7), int(TS_V>0), \
												alpha_hat_ts[age14], beta_hat_ts[age14],adj_V, adj_pi])

				TS_A = TS_A_1
				TS_H = TS_H_1
				TS_DC = TS_DC_1


				
		return wave_data_10, wave_data_14
	@njit
	def generate_TS(U,N,policy_a_noadj,etas_noadj,policy_H_adj,policy_a_adj,policy_c_adj,policy_Xi_cov,policy_Xi_copi,policy_VF):
		TSALL_10 = np.zeros((int((int(R)-int(tzero))*N*2),17))
		TSALL_14 = np.zeros((int((int(R)-int(tzero))*N*2),17))
		wave_length = 4
		policies = (policy_a_noadj,etas_noadj,policy_H_adj,policy_a_adj,policy_c_adj,policy_Xi_cov,policy_Xi_copi, policy_VF)
		k=int(0)
		for age in np.arange(int(tzero), int(R)):
			for i in range(N):
				length 		= 	int(age + wave_length+2) 
				W 			= 	sim_markov(P_E, P_stat, U[0, age, i])
				beta_hat_ts =	sim_markov(P_beta, beta_stat, U[2, age, i])
				alpha_hat_ts= 	sim_markov(P_alpha, alpha_stat, U[1, age, i])
				V_ushock_ts  = 	U[4, age, i]
				Pi_ushock_ts = 	U[3, age, i]
				DBshock 	= 	U[5, age, i,0]
				w10, w14 	= 	seriesgen(age, wave_length,W,beta_hat_ts,alpha_hat_ts,V_ushock_ts,Pi_ushock_ts,DBshock, *policies)
				TSALL_10[int(k),:] = w10
				TSALL_14[int(k),:] = w14
				k +=1

				# take the antithetic

				W 			= 	sim_markov(P_E, P_stat, 1-U[0, age, i])
				beta_hat_ts =	sim_markov(P_beta, beta_stat, 1-U[2, age, i])
				alpha_hat_ts= 	sim_markov(P_alpha, alpha_stat, 1-U[1, age, i])
				V_ushock_ts  = 	1-U[4, age, i]
				Pi_ushock_ts = 	1-U[3, age, i]
				DBshock 	= 	1-U[5, age, i,0]
				w10, w14 	= 	seriesgen(age, wave_length,W,beta_hat_ts,alpha_hat_ts,V_ushock_ts,Pi_ushock_ts,DBshock, *policies)
				TSALL_10[int(k),:] = w10
				TSALL_14[int(k),:] = w14
				k +=1


		return TSALL_10, TSALL_14


	def generate_TSDF(U,N,policy_a_noadj,etas_noadj,policy_H_adj,policy_a_adj,policy_c_adj,policy_Xi_cov,policy_Xi_copi,policy_VF):

		TSALL_10, TSALL_14 = generate_TS(U,N,policy_a_noadj,etas_noadj,policy_H_adj,policy_a_adj,policy_c_adj,policy_Xi_cov,policy_Xi_copi,policy_VF)


		TSALL_10_df = pd.DataFrame(TSALL_10)
		TSALL_14_df = pd.DataFrame(TSALL_14)

		col_list = list(['account_type', \
							  'Age', \
							  'wealth_fin', \
							  'wealth_real', \
							  'super_balance', \
							  'consumption', \
							  'Wages',\
							  'vol_cont', \
							  'vol_total',\
							  'risky_share', \
							  'house_adj', \
							  'risky_share_adj',\
							  'vol_cont_adj', \
							  'alpha_hat', \
							  'Beta_hat', \
							  'Adjustment_cost_v', \
							  'Adjustment_cost_pi'])
	

		TSALL_10_df.columns = col_list
		TSALL_14_df.columns = col_list
		
		return TSALL_10_df, TSALL_14_df

	return generate_TSDF


def gen_moments(TSALL_10_df, TSALL_14_df):

	age 	= np.arange(18, 65)
	main_df = pd.DataFrame(age)

	# age buckets are LHS closed RHS open
	# final age bucket is t = (58, 63] and hence age = (59, 64]
	# first age bucket is age = (19, 24] 
	age_buckets = np.arange(19, 65,5)

	
	keys_vars = set(TSALL_10_df.keys())
	excludes = set(['Adjustment_cost_v', \
				 'Adjustment_cost_pi', 'alpha_hat', 'Beta_hat'])

	TSALL_10_df.drop(excludes, axis = 1)    
	

	"""	for  key in keys_vars.difference(excludes):
			#print(key)
			main_df = pd.concat([main_df, pd.DataFrame( \
						np.transpose(TSALL[key][:,18:65]*100)). \
						add_prefix('{}'.format(key))], axis =1)


		# generate moments for wave 10
		main_df['id'] = main_df.index
		main_df = main_df.rename(columns ={0:'Age'})
		main_df = pd.wide_to_long(main_df, stubnames = \
					 list(keys_vars.difference(excludes)), i = "id", j = "memberid")   
	"""
	main_df = TSALL_10_df
	main_df['vol_cont_adj']= main_df['vol_cont_adj']
	main_df['risky_share_adj']= main_df['risky_share_adj']
	main_df['risky_share']= main_df['risky_share']
	main_df['vol_cont']=main_df['vol_cont']

	# adjust age so it is 'real age'
	#print(main_df['Age'])
	main_df['Age'] =main_df['Age']+ 1    

	#account_type_df  = pd.DataFrame(TSALL['account_type_all'])     
	#account_type_df['memberid'] = account_type_df.index 
	#account_type_df['memberid'].reset_index()
	#main_df = pd.merge(main_df, account_type_df, left_on = 'memberid', \
	#			 right_on = 'memberid', how = "left", validate = "m:1")  
	#main_df = main_df.rename(columns = {0: 'account_type'})

	means = main_df.groupby(pd.cut(main_df['Age'], age_buckets)).mean().add_prefix('mean_')  
	means = means.reset_index()   

	sds_all = main_df.groupby(pd.cut(main_df['Age'], \
				age_buckets)).std().add_prefix('sd_')
	sds_all = sds_all.reset_index().drop(['Age','sd_Age', \
				'sd_account_type'], axis =1)    

	sds_DB 	= main_df[main_df['account_type']==0].groupby(pd.cut \
				(main_df[main_df['account_type']==0]['Age'], age_buckets)) \
				.std().add_prefix('sd_').add_suffix('DB')
	sds_DB	= sds_DB.reset_index().drop(['Age','sd_AgeDB', \
				'sd_account_typeDB'], axis =1)

	sds_DC 	= main_df[main_df['account_type']==1].groupby(pd. \
				cut(main_df[main_df['account_type']==1]['Age'], age_buckets)). \
				std().add_prefix('sd_').add_suffix('DC')
	sds_DC  = sds_DC.reset_index().drop(['Age','sd_AgeDC', \
				 'sd_account_typeDC'], axis =1)

	corrlist = list(list(permutations(keys_vars.difference(excludes),2)))
	corrs_df = pd.DataFrame(means.index)  

	for corrs in corrlist:

		corrs_temp  = main_df.groupby(pd.cut(main_df['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp  = corrs_temp.reset_index()
		corrs_temp  = np.array(corrs_temp[corrs_temp.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)], \
							axis=1, inplace=False)

		# for DB only
		corrs_temp_DB  = main_df[main_df['account_type']==0] \
						 .groupby(pd.cut(main_df[main_df['account_type']==0]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DB  = corrs_temp_DB.reset_index()
		corrs_temp_DB  = np.array(corrs_temp_DB[corrs_temp_DB.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DB)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DB'], \
							axis=1, inplace=False)

		# for DC only
		corrs_temp_DC  = main_df[main_df['account_type']==1] \
						 .groupby(pd.cut(main_df[main_df['account_type']==1]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DC  = corrs_temp_DC.reset_index()
		corrs_temp_DC  = np.array(corrs_temp_DC[corrs_temp_DC.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DC)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DC'], \
							axis=1, inplace=False)


			
				
	corrs_df = corrs_df.add_prefix('corr_')

	moments = pd.concat([means.reset_index(), sds_all.reset_index(),\
				 corrs_df.reset_index(), sds_DB.reset_index()\
				 ,sds_DC.reset_index()  ], axis = 1)   
	
	moments_10 = moments.drop(['index'],axis = 1).add_suffix('_wave10')   

	main_df10 = main_df

	main_df10 = main_df10.add_suffix('_wave10')

	# now generate moments for wave 14

	# this method will need to change depending on meeting with ISA 

	main_df = TSALL_14_df
	main_df['vol_cont_adj']= main_df['vol_cont_adj']
	main_df['risky_share_adj']= main_df['risky_share_adj']
	main_df['risky_share']= main_df['risky_share']
	main_df['vol_cont']=main_df['vol_cont']

	# adjust age so it is 'real age'

	main_df['Age'] += 1    

	means = main_df.groupby(pd.cut(main_df['Age'], age_buckets)).mean().add_prefix('mean_')  
	means = means.reset_index() 

	sds_all = main_df.groupby(pd.cut(main_df['Age'], \
				age_buckets)).std().add_prefix('sd_')
	sds_all = sds_all.reset_index().drop(['Age','sd_Age', \
				'sd_account_type'], axis =1)    

	sds_DB 	= main_df[main_df['account_type']==0].groupby(pd.cut \
				(main_df[main_df['account_type']==0]['Age'], age_buckets))\
				.std().add_prefix('sd_').add_suffix('DB')
	sds_DB	= sds_DB.reset_index().drop(['Age','sd_AgeDB',\
				'sd_account_typeDB'], axis =1)

	sds_DC 	= main_df[main_df['account_type']==1].groupby(pd. \
				cut(main_df[main_df['account_type']==1]['Age'], age_buckets)). \
				std().add_prefix('sd_').add_suffix('DC')
	sds_DC  = sds_DC.reset_index().drop(['Age','sd_AgeDC',\
				 'sd_account_typeDC'], axis =1)

	corrlist = list(list(permutations(keys_vars.difference(excludes),2)))
	corrs_df = pd.DataFrame(means.index)  

	for corrs in corrlist:

		corrs_temp  = main_df.groupby(pd.cut(main_df['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp  = corrs_temp.reset_index()
		corrs_temp  = np.array(corrs_temp[corrs_temp.columns[1]] \
						.reset_index())[:,1]
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)], \
							axis=1, inplace=False)
		# for DB only
		corrs_temp_DB  = main_df[main_df['account_type']==0] \
						 .groupby(pd.cut(main_df[main_df['account_type']==0]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DB  = corrs_temp_DB.reset_index()
		corrs_temp_DB  = np.array(corrs_temp_DB[corrs_temp_DB.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DB)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DB'], \
							axis=1, inplace=False)

		# for DC only
		corrs_temp_DC  = main_df[main_df['account_type']==1] \
						 .groupby(pd.cut(main_df[main_df['account_type']==1]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DC  = corrs_temp_DC.reset_index()
		corrs_temp_DC  = np.array(corrs_temp_DC[corrs_temp_DC.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DC)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DC'], \
							axis=1, inplace=False)

			
				
	corrs_df 		= corrs_df.add_prefix('corr_')

	moments_14 		= pd.concat([means.reset_index(), sds_all.reset_index(),\
						corrs_df.reset_index(), sds_DB.reset_index()\
						,sds_DC.reset_index()  ], axis = 1) \
						.reset_index() \
						.add_suffix('_wave14') \
						.drop(['Age_wave14', 'index_wave14', 'level_0_wave14'],
							 axis= 1)

	main_df14 		= main_df
	main_df14		= main_df14.add_suffix('_wave14')

	main_horiz 		= pd.concat((main_df10, main_df14), axis =1)  
	
	# auto-correlation
	auto_corrs_df = pd.DataFrame(means.index)

	for keys in keys_vars.difference(excludes):
		corrs = list((keys+'_wave10', keys+'_wave14'))
		autocorrs_temp 	= main_horiz.groupby(pd.cut(main_horiz['Age_wave10'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		autocorrs_temp 	= autocorrs_temp.reset_index()
		autocorrs_temp  = np.array(autocorrs_temp[autocorrs_temp.columns[1]] \
						.reset_index())[:,1]
		auto_corrs_df 	= pd.concat([auto_corrs_df.reset_index(drop = True), \
							pd.DataFrame(autocorrs_temp)  ], axis = 1) 
		auto_corrs_df	=	auto_corrs_df.set_axis([*auto_corrs_df.columns[:-1], \
							 keys+'_autocorr'], \
							 axis=1, inplace=False)

	# generate conditional voluntary contribution 

	moments_10['mean_vol_cont_c_wave10']= moments_10['mean_vol_total_wave10']/moments_10['mean_vol_cont_adj_wave10']
	moments_14['mean_vol_cont_c_wave14']= moments_14['mean_vol_total_wave14']/moments_14['mean_vol_cont_adj_wave14']

	moments_10['mean_vol_cont_c_wave10']= moments_10['mean_vol_cont_c_wave10'].fillna(0)
	moments_14['mean_vol_cont_c_wave14']=moments_14['mean_vol_cont_c_wave14'].fillna(0)

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
						moments_male['corr_super_balance_risky_share_DC_wave14_male'],
						moments_male['corr_super_balance_risky_share_DB_wave14_male'],
						moments_female['corr_super_balance_risky_share_DC_wave14_female'],
						moments_female['corr_super_balance_risky_share_DB_wave14_female'],
						moments_male['corr_super_balance_risky_share_DC_wave10_male'],
						moments_male['corr_super_balance_risky_share_DB_wave10_male'],
						moments_female['corr_super_balance_risky_share_DC_wave10_female'],
						moments_female['corr_super_balance_risky_share_DB_wave10_female'],
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
						moments_male['sd_risky_shareDC_wave14_male'],
						moments_male['sd_risky_shareDB_wave14_male'],
						moments_female['sd_risky_shareDC_wave14_female'],
						moments_female['sd_risky_shareDB_wave14_female'],
						moments_male['sd_risky_shareDC_wave10_male'],
						moments_male['sd_risky_shareDB_wave10_male'],
						moments_female['sd_risky_shareDC_wave10_female'],
						moments_female['sd_risky_shareDB_wave10_female'],
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
						moments_male['mean_risky_share_wave14_male'],
						moments_female['mean_risky_share_wave14_female'],
						moments_male['mean_risky_share_wave10_male'],
						moments_female['mean_risky_share_wave10_female'],
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
						moments_female['mean_consumption_wave10_female']], axis =1)

	return moments_sorted 

if __name__ == '__main__':
	import os

	import seaborn as sns

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
		
		

		#og_male = pickle.load(open("/scratch/pv33/baseline_male.mod","rb")) 
		#og_female = pickle.load(open("/scratch/pv33/baseline_female.mod","rb")) 
		
		#og_male.accountdict 		= {}
		#og_male.accountdict[1] 		= 'DC'
		#og_male.accountdict[0] 		= 'DB'
		#og_female.accountdict 		= {}
		#og_female.accountdict[1] 		= 'DC'
		#og_female.accountdict[0] 		= 'DB'


		def create_plot(df,col1,col2, source, marker,color, ylim, ylabel):
			#df[col1] = df[col1].map(lambda x :inc_y0(x))
			#sns.set()
			 #list(set(df[source]))
			line_names =['wave14_data', 'wave10_data', 'wave14_sim', 'wave10_sim']
			linestyles=["-","-","-","-"]
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



			df_male =df_male.melt('Age_wave10', var_name= 'source', value_name = key)
			df_male['source'] = df_male['source'].str.replace(key+"_", "")
			df_female =df_female.melt('Age_wave10', var_name= 'source', value_name = key)
			df_female['source'] = df_female['source'].str.replace(key+"_", "")
			

			markers=['x', 'o', 'x', 'o']

			if col2 in normalise_list:
				df_male[col2] = df_male[col2].div(1000)
				df_female[col2] = df_female[col2].div(1000)

			figure, axes = plt.subplots(nrows=1, ncols=2)

			for name, marker, linestyle in zip(line_names, markers, linestyles):
						data_male = df_male.loc[df_male[source]==name]
						data_female = df_female.loc[df_female[source]==name]
						xs = list(data_male[col1])[0:18]
						ys = list(data_male[col2])
						p = axes[0].plot(xs, ys, marker=marker, color=col_dict[name], linestyle=linestyle,
									 label=name, linewidth=2)
						ys = list(data_female[col2])
						p = axes[1].plot(xs, ys, marker=marker, color=col_dict[name], linestyle=linestyle,
									label=name, linewidth=2)

			
			if isinstance(ylabel, str):
						ylabel = ylabel
			
			axes[0].set_title('Males')
			axes[0].set_xlabel('Age cohort')
			#axes[0].set_ylabel(ylabel)
			axes[0].set_ylim(ylim)
			axes[0].spines['top'].set_visible(False)
			axes[0].spines['right'].set_visible(False)
			axes[0].legend(loc='upper left', ncol=2)
			#plt.show()
			axes[1].set_title('Females')
			axes[1].set_xlabel('Age cohort')
		   #axes[1].set_ylabel(ylabel)
			axes[1].set_ylim(ylim)
			axes[1].spines['top'].set_visible(False)
			axes[1].spines['right'].set_visible(False)
			axes[1].legend(loc='upper left', ncol=2)
			#plt.tight_layout()
			#figure.size(10,10)
			figure.savefig("{}/{}.png".format(sim_id,col2), transparent=True)
		   

		#TSALL_10_male,TSALL_14_male  = genprofiles(og_male, N = 100)

		#pickle.dump(TSALL_10_male, open("/scratch/pv33/TSALL_10_male.mod","wb"))
		#pickle.dump(TSALL_14_male, open("/scratch/pv33/TSALL_14_male.mod","wb"))

		#TSALL_10_female,TSALL_14_female  =genprofiles(og_female, N = 100)


		#pickle.dump(TSALL_10_female, open("/scratch/pv33/TSALL_10_female.mod","wb"))
		#pickle.dump(TSALL_14_female, open("/scratch/pv33/TSALL_14_female.mod","wb"))

		og =pickle.load(open("base_model.md","rb"))

		# generate random numbers for the simulation

		U = np.random.rand(6,100,250,100)       

		# generate TS
		generate_TSDF     = genprofiles_operator(og)
		TS1, TS2 		= generate_TSDF(U,100, *og.policies)
		#TSALL_10_female, TSALL_14_female= generate_TS(U,N = 250)
		#moments_male        = gen_moments(copy.copy(TS1),copy.copy(TS2))   
		#moments_female      = gen_moments(copy.copy(TS1),copy.copy(TS2)) 

		#TSALL_10_male, TSALL_14_male = pickle.load(open("TSALL_10_{}.mod".format(sim_id_m),"rb")), pickle.load(open("TSALL_14_{}.mod".format(sim_id_m),"rb"))
		#TSALL_10_female, TSALL_14_female = pickle.load(open("TSALL_10_{}.mod".format(sim_id_f),"rb")), pickle.load(open("TSALL_14_{}.mod".format(sim_id_f),"rb")) 

		 
		moments_male = gen_moments(TS1,TS2)
		#moments_male = pd.read_csv('moments_male_old.csv')
		moments_female = gen_moments(TS1,TS2)

		moments_female = moments_female.add_suffix('_female')
		moments_male = moments_male.add_suffix('_male')
		
		moments_male.to_csv("/scratch/pv33/moments_male.csv") 
		moments_female.to_csv("/scratch/pv33/moments_female.csv") 

		moments_sorted 	= sortmoments(moments_male, moments_female)
		

		moments_sorted.to_csv("{}/moments_sorted.csv".format(sim_id))  

		moments_sorted = pd.concat([moments_male["Age_wave10_male"].reset_index().iloc[:,1], moments_sorted], axis =1)  

		moments_sorted = moments_sorted.rename(columns = {'Age_wave10_male':'Age_wave10'})
		
		moments_data = pd.read_csv('moments_data.csv')
		moments_data = moments_data.drop('Unnamed: 0', axis=1)   

		moments_data.columns = moments_sorted.columns



		age 		= np.arange(18, 65) # PROBABLY SHOULD GET RID OF AGE MAGIC NUMBERS HERE 

		plot_keys_vars 	= [	'mean_account_type',
							'corr_super_balance_risky_share_adj_DC',
							'corr_super_balance_risky_share_adj_DB',
							'corr_super_balance_risky_share_DC',
							'corr_super_balance_risky_share_DB',
							'corr_super_balance_vol_cont_adj_DC',
							'corr_super_balance_vol_cont_adj_DB',
							'corr_vol_total_super_balance_DC',
							'corr_vol_total_super_balance_DB',
							'corr_consumption_wealth_real',
							'sd_risky_shareDC',
							'sd_risky_shareDB',
							'sd_vol_totalDC',
							'sd_vol_totalDB',
							'sd_super_balanceDC',
							'sd_super_balanceDB',
							'sd_wealth_real',
							'sd_wealth_fin',
							'mean_risky_share',
							'mean_vol_cont_adj',
							'mean_vol_total',
							'mean_vol_cont_c',
							'mean_wealth_real',
							'mean_wealth_fin',
							'mean_super_balance',
							'mean_consumption', 
							'sd_consumption']




		plot_autocors = ['wealth_real_autocorr',
							'wealth_fin_autocorr',
							'consumption_autocorr']
		

		# variables with y axis 0:1000
		# real wealth, fin wealth super balance
		axis_style_list_ylim = {
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
							'mean_risky_share':(0,1),
							'sd_risky_shareDC':(0,1),
							'sd_risky_shareDB':(0,1),
							'sd_vol_totalDC':(0,15),
							'sd_vol_totalDB':(0,15),
							'mean_vol_total': (0,15),
							'mean_vol_cont_c': (0,15),
		# varibales with y axis -.5 to 1(correlations)
							'corr_super_balance_risky_share_adj_DC':(-.8,1),
							'corr_super_balance_risky_share_adj_DB':(-.8,1),
							'corr_super_balance_risky_share_DC':(-.8,1),
							'corr_super_balance_risky_share_DB':(-.8,1),
							'corr_super_balance_vol_cont_adj_DC':(-.8,1),
							'corr_super_balance_vol_cont_adj_DB':(-.8,1),
							'corr_vol_total_super_balance_DC':(-.8,1),
							'corr_vol_total_super_balance_DB':(-.8,1),
							'corr_consumption_wealth_real':(-.8,1)}

		axis_label_list = {
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
							'mean_risky_share': 'Share of Unisuper balance in risky assets',
							'sd_risky_shareDC':	'SD: Share of Unisuper balance in risky assets among DCer',
							'sd_risky_shareDB':	'SD: Share of Unisuper balance in risky assets among DBer',
							'sd_vol_totalDC':'SD: total voluntary contribution among DCer',
							'sd_vol_totalDB':'SD: total voluntary contribution among DBer',
							'mean_vol_total': 'Total voluntary contribution',
							'mean_vol_cont_c':'Total voluntary contribution (among contributors)',
		# varibales with y axis -.5 to 1(correlations)
							'corr_super_balance_risky_share_adj_DC':'CORR: UniSuper balane and non−default inv among DCer',
							'corr_super_balance_risky_share_adj_DB':'CORR: UniSuper balane and non−default inv among DBer',
							'corr_super_balance_risky_share_DC':'CORR: UniSuper balane and risky share among DCer',
							'corr_super_balance_risky_share_DB':'CORR: UniSuper balane and risky share among DCer',
							'corr_super_balance_vol_cont_adj_DC':'CORR: UniSuper balane and +vc among DCer',
							'corr_super_balance_vol_cont_adj_DB':'CORR: UniSuper balane and +vc among DBer',
							'corr_vol_total_super_balance_DC':'CORR: UniSuper balane and vc among DCer',
							'corr_vol_total_super_balance_DB':'CORR: UniSuper balane and vc among DCer',
							'corr_consumption_wealth_real':'CORR: Consumption and real wealth'}


		#excludes 	= set(['account_type_all'])

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

		for  key in plot_keys_vars:

			#Assets 
			var_data = moments_data[[key+'_wave10_male',key+'_wave10_female',key+'_wave14_male',key+'_wave14_female']]
			var_data = var_data.add_suffix('_data')
			var_data.iloc[8,2] = float('nan')
			var_data.iloc[8,3] =float('nan')  


			var_sim = moments_sorted[[key+'_wave10_male',key+'_wave10_female',key+'_wave14_male',key+'_wave14_female']]
			var_sim = var_sim.add_suffix('_sim')
			var_sim.iloc[8,2] = float('nan')
			var_sim.iloc[8,3] =float('nan')  


			var_grouped = pd.concat([cohort_labels, var_data, var_sim], axis =1)
			
	 
			ylim = axis_style_list_ylim[key]

			create_plot(var_grouped, 'Age_wave10', key, 'source', marker='s',color='darkblue', ylim = ylim, ylabel = axis_label_list[key])


