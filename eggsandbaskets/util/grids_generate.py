"""
Module contains function to generate 
state-space points at which 
functions are interpolated 
for a housing model class"""

# Import packages

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from numba import njit, prange, guvectorize, jit
from sklearn.utils.extmath import cartesian

import gc


def generate_points(comm,  my_age, og, path, scratch = True):

	amort_rate = og.functions.amort_rate
	b_prime = og.functions.b_prime

	V,Pi, M,H, Q, A = og.grid1d.V, og.grid1d.Pi, og.grid1d.M, og.grid1d.H, og.grid1d.Q, og.grid1d.A
	v_S, v_E = og.parameters.v_S, og.parameters.v_E
	A_DC = og.grid1d.A_DC
	A_min = og.parameters.A_min

	tau_housing = og.parameters.tau_housing
	delta_housing = og.parameters.delta_housing

	R = og.parameters.R
	tzero  = og.parameters.tzero
	yvec  = og.functions.yvec
	E  = og.st_grid.E
	r = og.parameters.r
	

	r_H  = og.parameters.r_H
	r_l  = og.parameters.r_l
	l  = og.parameters.l
	beta_m  = og.parameters.beta_m
	kappa_m = og.parameters.kappa_m

	alpha_bar = og.parameters.alpha_bar
	beta_bar = og.parameters.beta_bar
	alpha_hat = og.st_grid.alpha_hat
	beta_hat  = og.st_grid.beta_hat

	EBA_P = og.cart_grids.EBA_P

	X_all_ind = og.BigAssGrids.X_all_ind_f()
	Q_DC_shocks = og.cart_grids.Q_DC_shocks
	r_m_prime = og.cart_grids.r_m_prime
	X_all_hat_ind = og.BigAssGrids.X_all_hat_ind_f()

	#grid sizes
	grid_size_A, grid_size_DC, grid_size_H,\
	grid_size_Q, grid_size_M, grid_size_C = og.parameters.grid_size_A,\
									  og.parameters.grid_size_DC,\
									  og.parameters.grid_size_H,\
									  og.parameters.grid_size_Q,\
									  og.parameters.grid_size_M,\
									  og.parameters.grid_size_C
	grid_size_W, grid_size_alpha, grid_size_beta\
								=   int(og.parameters.grid_size_W),\
									int(og.parameters.grid_size_alpha),\
									int(og.parameters.grid_size_beta)

	@njit
	def gen_prob_EBA(i):

		""" 
		Give t+1 joint probability
		of E, alpha and beta state 
		given state i at time t

		Where indexes the cart. prod. 
		of E x alpha x beta grids 
		"""

		E_ind       = X_all_hat_ind[i][1]
		alpha_ind   = X_all_hat_ind[i][2]
		beta_ind    = X_all_hat_ind[i][3]
		Probs       = EBA_P[E_ind, alpha_ind, beta_ind,:]

		return Probs

	@njit
	def gen_EBA_P_mat():
		""" 
		For each state in the grid X_all_hat_vals,
		generate t+1 joint probability over 
		E, alpha and beta conditioned on 
		time t state i  in X_all_hat_vals

		""" 

		EBA_P_mat   = np.empty((len(X_all_hat_ind), len(EBA_P[0, 0, 0,:])))

		for i in prange(len(X_all_hat_ind)):
			probs = gen_prob_EBA(i)

			EBA_P_mat[i,:] = probs
			"""
			
			0 - DB/DC
			1 - E     (previous period)
			2 - alpha (previous period)
			3 - beta  (previous period)
			4 - Pi    (previous period)
			5 - A *before returns* at T_R
			6 - A_DC (before returns) taken into T_R
			7 - H at T_R (coming into state, BEFORE depreciation)
			8 - Q at T_R (previous period)
			"""
		return EBA_P_mat

	@njit(cache=True) 
	def gen_x_prime_vals(pi_ind, ADC_ind, q_ind, m_ind):
		"""
		0 - DB/DC
		1 - E     (previous period)
		2 - alpha (previous period)
		3 - beta  (previous period)
		4 - Pi    (previous period)
		5 - A *before returns* at T_R
		6 - A_DC (before returns) taken into T_R
		7 - H at T_R (coming into state, BEFORE depreciation)
		8 - Q at T_R (previous period)
		9 - M before returns taken into period 

		Returns
		------

		Note:
		M_prime is end of t period mortgage leverage but at time t period prices 
		"""
		r_share = Pi[pi_ind]
		ADC_in = A_DC[ADC_ind]
		q_in  = Q[q_ind]
		m_in  = M[m_ind]

		q_t_arr = np.full(len(Q_DC_shocks[:,2]),q_in)
		r_H_arr = np.full(len(Q_DC_shocks[:,2]),r_H)
		Q_prime = (1+r_H_arr + Q_DC_shocks[:,2])*q_t_arr

		risky_share_arr = np.full(len(Q_DC_shocks[:,2]),r_share)

		A_DC_returns = (1-risky_share_arr)*Q_DC_shocks[:,0] +\
								risky_share_arr*Q_DC_shocks[:,1]

		A_DC_prime  = A_DC_returns*np.full(len(Q_DC_shocks[:,2]),ADC_in)
		M_prime = (1+r_m_prime)*m_in*q_t_arr/(1-delta_housing) * Q_prime #check this 

		return np.column_stack((A_DC_prime,Q_prime,M_prime, r_m_prime))


	#@njit
	def gen_x_prime_array():

		X_prime_vals = np.empty((int(len(Pi)*len(A_DC)*len(Q)*len(M)),\
									len(Q_DC_shocks[:,2]), 4))

		X_prime_grid = cartesian([np.arange(len(Pi)),\
									 np.arange(len(A_DC)),\
									np.arange(len(Q)),\
									np.arange(len(M))])

		for i in range(len(X_prime_grid)):
			X_prime_vals[i,:] = gen_x_prime_vals(X_prime_grid[i,0],\
													X_prime_grid[i,1],\
													X_prime_grid[i,2],\
													X_prime_grid[i,3])


		del X_prime_grid 
		gc.collect()
		return X_prime_vals.reshape(len(Pi),\
									len(A_DC),\
									len(Q),len(M),len(Q_DC_shocks[:,2]),\
									4)

	@njit
	def gen_alph_beta(i):
		alpha_hs = alpha_hat[X_all_ind[i][2]]
		beta = beta_hat[X_all_ind[i][3]]

		return np.array([alpha_hs, beta])

	@njit
	def gen_alpha_beta_array():
		X_all_ind_W_vals = np.empty((len(X_all_ind), 2))

		for i in prange(len(X_all_ind)):

			"""vals in X_all_ind_W_vals are:

			0 - alpha
			1 - beta"""

			X_all_ind_W_vals[i,:] = gen_alph_beta(i)

		return X_all_ind_W_vals

	@njit
	def gen_wage_vec(j):
		#wage_vector = np.empty((int(R-tzero), len(X_all_ind[:,1])))

		#for j in prange(int(R-tzero)):
		wage_vector = yvec(int(j), E[X_all_ind[:,1]] )

		return wage_vector

   # def gen_RHS_points(comm):

		""" 
		Generates points for each t over which 
		the policy functions are evaluated to 
		generate points for the RHS of the 
		Euler equation  at each iteration 
		(see equation x in paper)
		
		Parameters
		----------

		Returns
		----------
		points_noadj_vec:  flat 9D array
		points_adj_vec:    flat 9D array
		points_rent_vec:    flat 9D array
		A_prime:            flat 9D array

		"""

	def _gen_points_for_age(j):

	#for j in range(int(R-tzero)):

		"""
		Each element of X_all_ind[i] is a 
		state index as follows:

		0 - DB/DC
		1 - E     
		2 - alpha 
		3 - beta  
		4 - V 
		5 - Pi    
		6 - A
		7 - A_DC *note this is A_DC at time t after returns from t-1*
		8 - H 
		9 - Q 
		10- M

		""" 

		acc_ind = np.full(len(X_all_ind[:,4]), int(0))
		v_Sv = np.full(len(X_all_ind[:,4]), v_S)
		v_Ev = np.full(len(X_all_ind[:,4]), v_E)
		v = V[X_all_ind[:,4]]

		# 1- total contribute rate as % of wage for all states 
		contrat = np.ones(len(X_all_ind[:,4])) - v -v_Sv - v_Ev

		pi = Pi[X_all_ind[:,5]]
		m = M[X_all_ind[:,10]] # this is leverage ratio wrt to time t house price 
		h = H[X_all_ind[:,8]]*(1-delta_housing)
		q = Q[X_all_ind[:,9]]
		m_val = h*m*Q[X_all_ind[:,9]]
		tau_housing_vec = np.full(len(X_all_ind[:,4]), tau_housing)

		wage = gen_wage_vec(j)

		points_noadj_vec, points_adj_vec, points_rent_vec\
		= np.empty((len(X_all_ind[:,1]),2)),\
			np.empty((len(X_all_ind[:,1]),2)),\
			np.empty((len(X_all_ind[:,1]),2))

		# total liquid wealth (cash in hand) for non-adjuster
		# and adjuster 
		points_noadj_vec[:,0] = A[X_all_ind[:,6]]*(1+r) + contrat*wage

		points_noadj_vec[:,0][points_noadj_vec[:,0]<=0] = A_min

		points_adj_vec[:,0] = points_noadj_vec[:,0] + q*h*(1-m)

		# next period DC assets (before returns)
		# (recall domain of policy functions from def of eval_policy_W)
		points_noadj_vec[:,1] = A_DC[X_all_ind[:,7]] + v*wage+ (v_S +v_E)*wage\
													* X_all_ind[:,0]
		points_adj_vec[:,1]= A_DC[X_all_ind[:,7]]+ v*wage + (v_S +v_E)*wage\
													* X_all_ind[:,0]
		# renter cash at hand (after mortgage deduction)
		points_rent_vec[:,0]= points_noadj_vec[:,0] + q*h - m_val                # should mortgage go here?
		points_rent_vec[:,0][points_rent_vec[:,0]<=0] = A_min

		points_rent_vec[:,1] = A_DC[X_all_ind[:,7]] + v*wage + (v_Sv +v_Ev)\
								* wage * X_all_ind[:,0]

		A_prime =  points_adj_vec[:,0] - m_val
		A_prime[A_prime<=0] = 1e-200 

		# reshape the adjustment points to wide
		# recall the points adj_vec are ordered accordint to
		# X_all_ind  
		points_adj_vec = points_adj_vec.reshape((1,grid_size_W,\
														grid_size_alpha,\
														grid_size_beta,\
														len(V),\
														len(Pi),\
														grid_size_A,\
														grid_size_DC,\
														grid_size_H,\
														grid_size_Q,\
														grid_size_M,2))
		#  change adjuster coordinates to:
		#  DB(0),E(1), Alpha(2), Beta(3), Pi(5), Q(9),\
		#  M(10), V(4), A(6), A_DC(7), H(8), 2D points(11)

		points_adj_vec = points_adj_vec.transpose((0,1,2,3,5,9,10,4,6,7,8,11))

		# reshape to:
		# Age, |DBx Ex Alpha x BetaxPi xQxM| x|VxAxA_DCxH|x2
		# recall each i \in |DBx Ex Alpha x BetaxPi xQxM|
		# adjuster function will be reshaped to a
		# function on Wealth x A_DC 
		points_adj_vec = points_adj_vec.reshape((int(1*grid_size_W*grid_size_alpha*
													grid_size_beta*len(Pi)*\
													grid_size_Q*grid_size_M),\
													int(grid_size_H*len(V)*\
													grid_size_A*\
													grid_size_DC),2))
		# reshape the renter points  
		points_rent_vec = points_rent_vec.reshape((1,grid_size_W,\
														grid_size_alpha,\
														grid_size_beta,\
														len(V),\
														len(Pi),\
														grid_size_A,\
														grid_size_DC,\
														grid_size_H,\
														grid_size_Q,\
														grid_size_M,2))
		# change renter coordinates to:
		#  DB(0), E(1), Alpha(2), Beta(3), Pi(5), Q(9),\
		#  V(4), A(6), A_DC(7), H(8), M(10), points(2)
		points_rent_vec =points_rent_vec.transpose((0,1,2,3,5,9,4,6,7,8,10,11))

		points_rent_vec = points_rent_vec.reshape((int(1*grid_size_W*grid_size_alpha*
													grid_size_beta*len(Pi)*\
													grid_size_Q),\
													int(grid_size_H*len(V)*\
													grid_size_A*grid_size_M*\
													grid_size_DC),2))

		# reshape the no adjustment points  
		points_noadj_vec = points_noadj_vec.reshape((1,grid_size_W,\
														grid_size_alpha,\
														grid_size_beta,\
														len(V),\
														len(Pi),\
														grid_size_A,\
														grid_size_DC,\
														grid_size_H,\
														grid_size_Q,\
														grid_size_M,2))

		# change noadj coordinates to:
		# DB(0), E(1), Alpha(2), Beta(3), Pi(5), H(8), Q(9),M(10)\
		#  V(4), A(6), A_DC(7), points(11)
		points_noadj_vec = points_noadj_vec.transpose((0,1,2,3,5,8,9,10,4,6,7,11))
		points_noadj_vec = points_noadj_vec.reshape((int(1*grid_size_W*grid_size_alpha*
											grid_size_beta*len(Pi)*\
											grid_size_H*grid_size_Q*grid_size_M),\
											int(len(V)*
											grid_size_A*
											grid_size_DC),2))
		del wage
		gc.collect()

		return points_noadj_vec, points_adj_vec, points_rent_vec, A_prime


	for j in range(len(my_age)):
		age = int(my_age[j])
		#print(age)
		points_noadj_vec, points_adj_vec,points_rent_vec,A_prime =  _gen_points_for_age(age)

		np.savez_compressed(path + "/grid_modname_{}_age_{}".format(og.mod_name, age),\
									points_noadj_vec  = points_noadj_vec,\
									points_adj_vec = points_adj_vec,\
									points_rent_vec = points_rent_vec,\
									b_A_prime = A_prime)
			#X_prime_vals = gen_x_prime_array()
			#np.savez_compressed(path+"/grid_modname_{}_genfiles".format(og.mod_name),X_prime_vals = X_prime_vals)
			#del X_prime_vals
		del points_noadj_vec, points_adj_vec, points_rent_vec, A_prime

	del X_all_ind, X_all_hat_ind, og
	
	gc.collect()

	return None
