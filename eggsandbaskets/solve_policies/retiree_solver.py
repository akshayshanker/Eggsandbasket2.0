
"""
Module contains the HousingModel retiree solvers for Bateman et. al. (2021)

Functions: retiree_func_factory
			Generates the operators required to solve retiree
			policy functions using an instance of HousingModel class

Akshay Shanker
School of Economics
University of New South Wales
akshay.shanker@me.com

"""

import time
from interpolation.splines import extrap_options as xto
from interpolation.splines import eval_linear
from quantecon.optimize.root_finding import brentq
from interpolation import interp
#from interpolation.splines import UCGrid, CGrid, nodes
from numba import njit
import numpy as np
import gc
import sys
from mpi4py import MPI

from util.helper_funcs import interp_as

eval_linear_c = eval_linear

def retiree_func_factory(og):
	"""Creates operator to solve retiree problem

	Parameters
	----------
	og : HousingModel
		 Instance of housing model class

	Returns
	-------
	gen_R_pol: function
			   Generates marginal utilities for retiring workers at age 64

	"""

	# Load functions
	u = og.functions.u
	uc = og.functions.ucnz
	uh = og.functions.uh
	uh_inv = og.functions.uh_inv
	ces_c1 = og.functions.ces_c1
	ch_ser = og.functions.ch_ser

	u_vec, uc_vec = og.functions.u_vec, og.functions.uc_vec

	b, b_prime = og.functions_CD.b, og.functions_CD.b_prime
	y, DB_benefit = og.functions_CD.y, og.functions_CD.DB_benefit
	adj_p, adj_v, adj_pi = og.functions_CD.adj_p, og.functions_CD.adj_v,\
							 og.functions_CD.adj_pi

	# Parameters
	k = og.parameters.k
	phi_r = og.parameters.phi_r
	A_max_WE = og.parameters.A_max_WE
	delta_housing, alpha, beta_bar = og.parameters.delta_housing,\
										 og.parameters.alpha,\
										og.parameters.beta_bar
	
	tau_housing = og.parameters.tau_housing
	r, s, r_H = og.parameters.r, og.parameters.s, og.parameters.r_H
	r_l, beta_m, kappa_m = og.parameters.r_l, og.parameters.beta_m,\
							 og.parameters.kappa_m
	l = og.parameters.l
	alpha_housing = og.st_grid.alpha_housing

	# Grids
	Q_shocks_r, Q_shocks_P = og.st_grid.Q_shocks_r, og.st_grid.Q_shocks_P
	Q_DC_shocks, Q_DC_P = og.cart_grids.Q_DC_shocks, og.cart_grids.Q_DC_P
	r_m_prime = og.cart_grids.r_m_prime
	X_QH_R = og.interp_grid.X_QH_R

	H_R, HR_Q = og.grid1d.H_R, og.cart_grids.HR_Q
	A, A_DC, Q, H, M, W_R = og.grid1d.A, og.grid1d.A_DC, og.grid1d.Q,\
								 og.grid1d.H, og.grid1d.M, og.grid1d.W_R
	A_R, H_Q, A_Q_R, W_Q_R = og.grid1d.A_R, og.cart_grids.H_Q,\
								 og.cart_grids.A_Q_R, og.interp_grid.W_Q_R
	E, P_E, P_stat = og.st_grid.E, og.st_grid.P_E, og.st_grid.P_stat
	Pi = og.grid1d.Pi
	M = og.grid1d.M

	A_min, C_min, C_max, \
		H_min, A_max_R = og.parameters.A_min, og.parameters.C_min, og.parameters.C_max,\
		og.parameters.H_min, og.parameters.A_max_R
	H_max = og.parameters.H_max
	C_max = og.parameters.C_max
	C_min = og.parameters.C_min

	X_all_hat_ind_func = og.BigAssGrids.X_all_hat_ind_f

	X_cont_R, X_R_contgp,\
		X_H_R_ind,\
		X_RC_contgp, X_R_cont_ind = og.interp_grid.X_cont_R, og.interp_grid.X_R_contgp,\
		og.cart_grids.X_H_R_ind,\
		og.interp_grid.X_RC_contgp,\
		og.cart_grids.X_R_cont_ind

	grid_size_A, grid_size_DC,\
		grid_size_H, grid_size_Q,\
		grid_size_M, grid_size_C = og.parameters.grid_size_A,\
		og.parameters.grid_size_DC,\
		og.parameters.grid_size_H,\
		og.parameters.grid_size_Q,\
		og.parameters.grid_size_M,\
		og.parameters.grid_size_C
	grid_size_HS = og.parameters.grid_size_HS

	T, tzero, R = og.parameters.T, og.parameters.tzero, og.parameters.R

	acc_ind = og.acc_ind[0]


	@njit
	def uc_inv(uc,s,alpha):
		""" Inverts MUC for housing services

		Parameters
		----------
		uc: float64
			 marginal utility of consumption 
		s:  float64
			 housing services
		alpha: float64
				housing share in CES 

		Returns
		-------
		c: float64
			consumption 
		"""

		args = (s,alpha,uc)
		if ces_c1(C_min,s, alpha, uc)<0:
			return C_min
		elif ces_c1(C_max,s, alpha, uc)>0:
			return C_max
		elif ces_c1(C_min,s, alpha, uc)*ces_c1(C_max,s, alpha, uc)<0:
			return brentq(ces_c1, C_min, C_max, args = args, disp=False)[0]
		else:
			return C_min


	@njit
	def interp_adj(a_adj, c_adj, wealth_endgrid, extrap = True):
		"""Reshapes and interpolates policy functions
			for housing adjusters on endogenous net wealth
			grid """

		# Empty arrays to fill with interpolated values of policies. 
		# Recall that state is net wealth, house price and mortage 
		a_adj_uniform = np.empty((grid_size_Q, grid_size_A))
		H_adj_uniform = np.empty((grid_size_Q, grid_size_A))
		c_adj_uniform = np.empty((grid_size_Q, grid_size_A))

		a_adj_bar = np.transpose(a_adj.reshape(grid_size_H,
											   grid_size_Q ))
		wealth_endgrid_bar = np.transpose(wealth_endgrid.reshape(grid_size_H,\
												grid_size_Q))
		c_adj_bar = np.transpose(c_adj.reshape(grid_size_H, \
												grid_size_Q))

		for i in range(len(wealth_endgrid_bar)):

			wealthbar_c = wealth_endgrid_bar[i]
			A_bar_c = a_adj_bar[i]
			H_c = H
			C_c = c_adj_bar[i]
			wealth_x = wealthbar_c[~np.isnan(wealthbar_c)]

			assts_x = np.take(A_bar_c[~np.isnan(wealthbar_c)],
							  np.argsort(wealth_x))
			cons_x = np.take(C_c[~np.isnan(wealthbar_c)],
							 np.argsort(wealth_x))



			h_x = np.take(H_c[~np.isnan(wealthbar_c)],
						  np.argsort(wealth_x))

			wealth_x_sorted = np.sort(np.copy(wealth_x))
			h_x[wealth_x_sorted <= A_min] = H_min


			if extrap:
				c_adj_uniform[i] = interp_as(wealth_x_sorted, cons_x, W_R)

				c_adj_uniform[i][c_adj_uniform[i] <= C_min] = C_min
				c_adj_uniform[i][c_adj_uniform[i] > C_max] = C_max
				a_adj_uniform[i] = interp_as(wealth_x_sorted, assts_x, W_R)

				a_adj_uniform[i][a_adj_uniform[i] <= A_min] = A_min

				H_adj_uniform[i] = interp_as(wealth_x_sorted, h_x, W_R)
				H_adj_uniform[i][H_adj_uniform[i] <= H_min] = H_min
			else:
				c_adj_uniform[i] = np.interp(W_R, wealth_x_sorted, cons_x)

				c_adj_uniform[i][c_adj_uniform[i] <= C_min] = C_min
				c_adj_uniform[i][c_adj_uniform[i] > C_max] = C_max

				a_adj_uniform[i] = np.interp(W_R, wealth_x_sorted, assts_x)

				a_adj_uniform[i][a_adj_uniform[i] <= A_min] = A_min

				H_adj_uniform[i] = np.interp(W_R, wealth_x_sorted, h_x)
				H_adj_uniform[i][H_adj_uniform[i] <= H_min] = H_min

			H_adj_uniform[i][0] = H_min

		return np.reshape(a_adj_uniform, (grid_size_Q, grid_size_A)),\
				np.reshape(c_adj_uniform, (grid_size_Q, grid_size_A)),\
				np.reshape(H_adj_uniform, (grid_size_Q, grid_size_A))

	@njit
	def interp_no_adj(assets_endgrid_1, cons_1):
		""" Reshapes and interps the policy functions
			for housing non-adjusters on endogenous
			asset grid"""

		assets_endgrid = assets_endgrid_1.reshape(grid_size_A,
												  grid_size_H
												  * grid_size_Q * grid_size_M)
		assets_endgrid = np.transpose(assets_endgrid)

		cons_reshaped = cons_1.reshape(grid_size_A,
									   grid_size_H
									   * grid_size_Q * grid_size_M)
		cons_reshaped = np.transpose(cons_reshaped)

		assets_uniform = np.zeros(
			(grid_size_H * grid_size_Q * grid_size_M, grid_size_A))

		cons_uniform = np.zeros(
			(grid_size_H * grid_size_Q * grid_size_M, grid_size_A))

		for i in range(len(assets_uniform)):

			# Interp_as next period assets on current period endogenous
			# grid of assets
			assets_uniform[i] = interp_as(
				np.sort(
					assets_endgrid[i]), np.take(
					A_R, np.argsort(
						assets_endgrid[i])), A_R)

			assets_uniform[i][assets_uniform[i] < 0] = A_min

			# Interp_as consumption at t on current period endogenous grid of
			# assets
			cons_uniform[i] = interp_as(
				np.sort(
					assets_endgrid[i]), np.take(
					cons_reshaped[i], np.argsort(
						assets_endgrid[i])), A_R)

			cons_uniform[i][cons_uniform[i] < 0] = C_min

		# Re-shape interpolated policies on time t state
		a_noadj_1 = np.transpose(assets_uniform)
		a_noadj = np.reshape(np.ravel(a_noadj_1),
							 (grid_size_A,
							  grid_size_H,
							  grid_size_Q,
							  grid_size_M))

		c_noadj_1 = np.transpose(cons_uniform)
		c_noadj = np.reshape(np.ravel(c_noadj_1),
							 (grid_size_A,
							  grid_size_H,
							  grid_size_Q,
							  grid_size_M))

		return a_noadj, c_noadj

	@njit
	def rent_FOC(c, s, q):
		""" First order condition for renters.

			Zero this condition is optimal consumption given housing services.
			"""

		RHS = uh(c, s, alpha_housing) / (q * phi_r)

		return c - uc_inv(RHS, s, alpha_housing)

	@njit
	def gen_rent_pol():
		""" Generate renter aux. cons. policy on grid of time t housing services
			and time t house prices"""

		cons = np.zeros(len(H_Q))

		for i in range(len(H_Q)):
			cons[i] = brentq(rent_FOC, C_min, C_max, \
								args=(H_Q[i, 0], H_Q[i, 1]), disp = False)[0]

		return cons

	@njit
	def liq_rent_FOC(a_prime, cons, h, q, UC_prime,t):

		points = np.array([a_prime, H_min, q, 0])
		UC_prime_RHS = eval_linear_c(X_cont_R, UC_prime, points, xto.NEAREST)

		return uc(cons, h, alpha_housing) - UC_prime_RHS

	@njit
	def eval_rent_pol(UC_prime, t):

		a_end_1 = np.zeros(len(HR_Q))

		for i in range(len(HR_Q)):

			c_t = phi_r * HR_Q[i, 1] * HR_Q[i, 0] * \
				(1 - alpha_housing) / alpha_housing

			rent_focargs = (c_t, HR_Q[i, 0], HR_Q[i, 1], UC_prime, t)

			if liq_rent_FOC(A_min, *rent_focargs) * \
					liq_rent_FOC(A_max_WE, *rent_focargs) < 0:
				a_prime_1 = brentq(
					liq_rent_FOC, A_min, A_max_WE, args=rent_focargs, disp = False)[0]
			elif liq_rent_FOC(A_min, *rent_focargs) > 0:
				a_prime_1 = A_min
			elif liq_rent_FOC(A_max_WE, *rent_focargs) < 0:
				a_prime_1 = A_max_WE
			else:
				a_prime_1 = np.nan
				a_end_1[i] = np.nan

			a_end_1[i] = c_t + a_prime_1 + HR_Q[i, 1] * phi_r * HR_Q[i, 0]

		a_end = np.transpose(a_end_1.reshape((int(grid_size_HS), len(Q))))

		h_prime_func = np.zeros((len(Q), len(A)))

		for i in range(len(Q)):

			h_prime_func[i, :] = interp_as(np.sort(a_end[i][a_end[i] != np.nan]), np.take(
				H_R[a_end[i] != np.nan], np.argsort(a_end[i][a_end[i] != np.nan])), W_R)

			h_prime_func[i, :][h_prime_func[i, :] <= 0] = H_min

		return np.transpose(h_prime_func)

	@njit
	def HA_FOC(x_prime,
			   h,
			   q,
			   m_prime_func,
			   UF_prime,
			   UC_prime,
			   UC_prime_H,
			   UC_prime_M,
			   t,
			   ret_cons=False,
			   ret_mort=False):

		""" Function f(x) where x | f(x) = 0 is interior solution
		for a_t+1 given i) H_t where housing is adjusted and
		ii) mortgage repayment is constrained optimal

		Solutution to equation x in paper

		Parameters
		----------
		x_prime:            float64
							 a_t+1 next period liquid
		h:                  float64
							 H_t
		q:                  float64
							 P_t house price
		m:                  float64
							 time t mortgage leverage wrt time t price
		mort_func:          4D array
							 time t+1 mortgage leverage of unconstrainted
							 adjustment function of
							 a_t+1, h_t, q_t, c_t
							 leverage wrt to time t price 
		t_prime_funcs:      6-tuple
							 next period policy functions

		t:                  int
							 Age
		Returns
		-------

		Euler error:        float64

		"""

		m_prime, c_t = eval_c_mort(x_prime, h, q,
								   m_prime_func,
								   UC_prime, t)

		points = np.array([x_prime,h,q, m_prime])
		UC_prime_RHS  = max(1e-250, eval_linear_c(X_cont_R, UC_prime, points, xto.NEAREST))
		UC_prime_H_RHS  = max(1e-250,eval_linear_c(X_cont_R, UC_prime_H, points, xto.NEAREST))
		
		c_t = max(C_min, uc_inv(max(1e-300, UC_prime_RHS), h, alpha_housing))

		RHS = uc(c_t, h, alpha_housing) * q * (1 + tau_housing)\
			- UC_prime_H_RHS

		# return equation x in paper

		if ret_cons == True:
			return c_t

		elif ret_mort == True:
			return m_prime
		else:
			return uh(c_t, h, alpha_housing) - RHS

	@njit
	def H_FOC(c, x_prime,
			  h,
			  q,
			  UF_prime,
			   UC_prime,
			   UC_prime_H,
			   UC_prime_M, 
			   t,
			  ret_mort=False):

		""" Function f(x) where x | f(x) = 0  given x_t+1
		is interior solution
		for c_t given i) H_t where housing is adjusted and
		ii) mortgage repayment is constrained optimal

		Euler for liquid assets not interior.
		Note if Euler for liquid assetts not
		interior, then mortgage must be binding
		(see equation x in paper)

		Equation x in paper

		Parameters
		----------
		c:                  float64

		x_prime:            float64
							 a_t+1 next period liquid
		h:                  float64
							 H_t
		q:                  float64
							 P_t house price
		m:                  float64
							 time t mortgage liability
		t_prime_funcs:      6-tuple
							 next period policy functions

		t:                  int
							 Age
		Returns
		-------

		Euler error:        float64

		"""

		pointsf = np.array([x_prime,h,q, 0])
		UC_prime_RHSf  = max(1e-250, eval_linear_c(X_cont_R, UC_prime, \
											pointsf, xto.NEAREST))
		UC_prime_H_RHSf  = max(1e-250,eval_linear_c(X_cont_R, UC_prime_H,\
											 pointsf, xto.NEAREST))
		UC_prime_M_RHSf  = max(1e-250,eval_linear_c(X_cont_R, UC_prime_M,\
											 pointsf, xto.NEAREST))


		if UC_prime_H_RHSf < UC_prime_M_RHSf:
			m_prime = 0

		else:
			m_prime = M[-1]

		points = np.array([x_prime,h,q, m_prime])

		UC_prime_RHS  = max(1e-250,eval_linear_c(X_cont_R, UC_prime,\
													 points, xto.NEAREST))
		UC_prime_H_RHS  = max(1e-250,eval_linear_c(X_cont_R, UC_prime_H,\
													 points, xto.NEAREST))

		RHS = uc(c, h, alpha_housing) * q * (1 + tau_housing) - UC_prime_H_RHS

		if ret_mort:
			return m_prime

		else:
			return uh(c, h, alpha_housing) - RHS

	@njit
	def mort_FOC(m, x_prime,
				 h,
				 q,
				 UC_prime,
				 UC_prime_M, t):
		""" FOC for interior mortage decision. Mortgage decision 
			is t+1 mortgage leverage in terms of time t prices and end of time 
			t period housing stock (without t+1 depreciation taken out)"""

		points = np.array([x_prime, h,q,m])

		UC_prime_RHS = max(1e-250,eval_linear_c(X_cont_R, UC_prime, points,\
											  	xto.NEAREST))
		UC_prime_M_RHS = max(1e-250, eval_linear_c(X_cont_R, UC_prime_M, points,\
											 	xto.NEAREST))

		return UC_prime_RHS - UC_prime_M_RHS

	@njit
	def eval_c_mort(x_prime,
					h,
					q,
					m_prime_func,
					UC_prime, t):
		""" Evaluate consumption and
			mortgage with amort osntrained optimal
			mortgage and interior liquid asset FOC


		Equation x in paper

		Parameters
		----------
		c:                  float64
							 c_t
		x_prime:            float64
							 a_t+1 next period liquid
		h:                  float64
							 H_t
		q:                  float64
							 P_t house price
		m:                  float64
							 start of time t leverage in time t prices
		mort_func:          4D array
							 time t+1 M_t+1 (before interest)
							 given c,h, x_prime, q
		t_prime_funcs:      6-tuple
							 next period policy functions
		t:                  int
							 Age at time t
		Returns
		-------
		Euler error:    float64

		Note: mort_func is defined for given c, x_prime, h and q
				mort_func is mortgage given mort euler equation
				holding with equality

		"""

		m_prime = max(0, min(M[-1],\
						 np.interp(x_prime, A_R, m_prime_func)))

		points = np.array([x_prime, h,q,m_prime])

		UC_prime_RHS = max(1e-250, eval_linear_c(X_cont_R, UC_prime, points,\
														 xto.NEAREST))

		c_t = max(C_min, uc_inv(max(1e-250, UC_prime_RHS), h, alpha_housing))

		return c_t, m_prime

	@njit
	def gen_UC_RHS(t, 
				   x_prime,
				   h,
				   q,
				   m,
				   a_prime_noadj,
				   c_prime_noadj,
				   a_prime_adj,
				   c_prime_adj,
				   h_prime_adj,
				   h_prime_rent,
				   UF_dbprime,
				   UH_dbprime):

		""" At time t, Evaluates RHS value of Euler equation: marginal 
				utilities for t+1 discounted back to t

		Parameters
		----------

		t:              int
						 Age at time t
		x_prime:        float64
						 a_t+1 next period liquid asset
		h:              float64
						 H_t
		q:              float64
						 P_t house price
		m:              float64
						 m_t+1 mortgage leverage 
						(before t+1 interest in time t prices!)
		a_prime_noadj:  3D array
						 t+1 liquid asset function no-adj
						 defined on t+1 AxHxQ

		c_prime_noadj:  3D array
						 t+1 consumption function no-adj
						 defined on t+1 AxHxQ

		eta_prime_noadj:3D array
						 t+1 eta function no-adj
						 defined on t+1 AxHxQ
		a_prime_adj:    2D array
						 t+1 liquid assets adj
						 defined on QxW
		c_prime_adj:    2D array
						 t+1 liquid assets adj
						 defined on QxW
		h_prime_adj:    2D array
						 t+1 housing adj
						 defined on QxW
		UF_dbprime: 	2D array
						 t+1 value function 
						 defined on AxHxQxW

		Returns
		-------
		UC_prime_RHS:       float64
							 expected marginal utility of t+1 consumption 
		UC_prime_H_RHS:     float64
							 expected marginal utility of t+1 housing

		UC_prime_M_RHS:     float64
							 expected marginal utility of t+1 leverage
							 (leverage wrt to time t prices)


		Note: t+1 A is assets before returns, H is housing after dep.
		and W is wealth in hand after returns and sale of house next
		with period rices

		Note2: check intuition for why mort_prime_func plays no role
				in this function/
				what happens to the

		"""

		# STEP1: Evaluate t+ 1 states

		# array of t+1 house prices, mort. interest rates,
		# mortgage balances after interest and net wealth

		q_t_arr = np.full(len(Q_DC_shocks[:, 2]), q)

		Q_prime = q * (1 + r_H + Q_DC_shocks[:, 2])

		# t+1 leverage wrt. to t+1 house price after t+1 interest
		M_prime = r_m_prime * m * q_t_arr / ((1 - delta_housing) * Q_prime)
		M_prime_R = r_m_prime

		W_prime = (1 + r) * x_prime + Q_prime * np.full(len(Q_prime),
														(1 - delta_housing) * h) * (1 - M_prime)

		W_prime_renters = W_prime 

		W_prime[W_prime <= A_min] = A_min

		# t+1 states: A_t+1(1+r) - min mort payment payment, P_t+1, M_t+1 (1+r_m)
		# this is the state for the no  housing adjusters

		x_prime_array = np.full(len(Q_prime), x_prime * (1 + r))

		state_prime_R = np.column_stack((x_prime_array,
										 np.full(len(Q_prime),
												 (1 - delta_housing) * h),
										 Q_prime, M_prime))

		state_prime_R[:, 0][state_prime_R[:, 0] <= 0] = A_min
		state_prime_R[:, 0][np.isnan(state_prime_R[:, 0])] = A_min

		# t+1 states: P_t+1, M_t+1 (1+r_m), net wealth - min mort payment
		# this is the state for the   housing adjusters

		state_prime_RW = np.column_stack((Q_prime, W_prime))

		# t+1 states: t+1 states: P_t+1, M_t+1 (1+r_m),
		# net wealth - min mort payment - housing adjustment cost
		# this is state for renters

		state_prime_rent = np.column_stack((Q_prime, W_prime))


		state_prime_rent[:, 1][state_prime_rent[:, 1] < A_min] = A_min
		state_prime_rent[:, 1][np.isnan(state_prime_rent[:, 1])] = A_min
		# bequest value
		# should there be a (1+r) here that goes into the bequest function?
		A_prime = max(x_prime * (1 + r) + (1 - delta_housing)* h * q * (1 - m), A_min)

		# the bequest function should have the *Next* period house price and
		# mortgage repayment rate
		# STEP 2: evaluate multipliers
		#      eta_ind> 1 if NOT adjusting housing stock (cond on not renting)
		#      zeta_ind>1 if NOT making liquid saving (cond on not adjusting)

		# evaluate array of next period eta adjustment multipliers

		#v_prime_vals_adj = eval_linear_c(X_cont_R, UF_dbprime, state_prime_R, xto.NEAREST)
		#v_prime_vals_noadj = eval_linear_c(X_cont_R, UF_dbprime, state_prime_R, xto.NEAREST)

		# evaluate where adjustment occurs
		# adjustment occurs where someone eta >1
		# note mortgage default is not possible 

		# STEP 3: calc cons and a_prime for
		# adjusters if liq. saving made

		c_prime_adj_vals, a_prime_adj_vals,\
		h_prime_adj_vals = eval_linear_c(X_QH_R,
										   c_prime_adj,
										   state_prime_RW, xto.NEAREST),\
								eval_linear_c(X_QH_R,
											a_prime_adj,
											state_prime_RW, xto.NEAREST), \
								eval_linear_c(X_QH_R,
											h_prime_adj,
											state_prime_RW, xto.NEAREST)

		c_prime_adj_vals[c_prime_adj_vals <= C_min] = C_min
		c_prime_adj_vals[np.isnan(c_prime_adj_vals)] = C_min

		h_prime_adj_vals[h_prime_adj_vals <= H_min] = H_min
		h_prime_adj_vals[np.isnan(h_prime_adj_vals)] = H_min

		h_prime_adj_vals[state_prime_RW[:, 2] <= 0] = H_min
		h_prime_adj_vals[np.isnan(state_prime_RW[:, 2])] = H_min

		# STEP 4: calc cons and a_prime for non-adjusters

		c_prime_noadj_vals, a_prime_noadj_vals = eval_linear_c(
			X_cont_R, c_prime_noadj, state_prime_R, xto.NEAREST), eval_linear_c(
			X_cont_R, a_prime_noadj, state_prime_R, xto.NEAREST)

		c_prime_noadj_vals[c_prime_noadj_vals <= C_min] = C_min
		c_prime_noadj_vals[np.isnan(c_prime_noadj_vals)] = C_min
		c_prime_noadj_vals[state_prime_R[:, 0] <= 0] = C_min
		c_prime_noadj_vals[np.isnan(state_prime_R[:, 0])] = C_min

		h_prime_noadj_vals = np.full(len(Q_prime),
									 (1 - delta_housing) * h)
		h_prime_noadj_vals[h_prime_noadj_vals < 0] = H_min
		h_prime_noadj_vals[np.isnan(h_prime_noadj_vals)] = H_min

		# STEP 6: calculate mortgage payment and flag extra payment

		mort_pay_noadj = ((state_prime_R[:, 0] - c_prime_noadj_vals - a_prime_noadj_vals)
							) / Q_prime * np.full(len(Q_prime), (1 - delta_housing) * h)

		mort_dp_prime_noadj = M_prime - mort_pay_noadj

		mort_dp_prime_adj = (state_prime_RW[:, 1] - c_prime_adj_vals -
						  a_prime_adj_vals - h_prime_adj_vals * Q_prime *
						  (1 + tau_housing)) / Q_prime * h_prime_adj_vals
		#mort_dp_prime_adj = M_prime - mort_expay_adj



		# STEP 7: create vec of t+2 mortgage balances (before t+2 interest )

		# t+2 mortgage leverage wrt to t+1 prices before t+2 interest



		# STEP 8: combine all non-renting policies

		state_dp_prime_noadj = np.column_stack((a_prime_noadj_vals,\
												 h_prime_noadj_vals,\
												 Q_prime,\
												 mort_dp_prime_noadj))
		state_dp_prime_adj = np.column_stack((a_prime_adj_vals,\
											 h_prime_adj_vals,\
											 Q_prime,\
											 mort_dp_prime_adj))

		UF_dp_val_noadj = beta_bar * eval_linear_c(X_cont_R, \
												UF_dbprime,\
												state_dp_prime_noadj,\
												xto.NEAREST)

		UH_dp_val_norent = eval_linear_c(X_cont_R,\
													UH_dbprime,\
													state_dp_prime_noadj,\
													xto.NEAREST)

		UF_dp_val_adj = beta_bar * eval_linear_c(X_cont_R,\
												UF_dbprime,\
												state_dp_prime_adj,\
												 xto.NEAREST)

		V_noadj = u_vec(c_prime_noadj_vals, h_prime_noadj_vals, alpha_housing)\
					 + UF_dp_val_noadj
		V_adj = u_vec(c_prime_adj_vals, h_prime_adj_vals, alpha_housing)\
					 + UF_dp_val_adj

		eta_ind = V_noadj>V_adj

		mort_dp_prime = eta_ind*mort_dp_prime_noadj + (1-eta_ind)*mort_dp_prime_adj

		mort_dp_prime[mort_dp_prime < 0] = 0

		c_prime_val_norent = c_prime_adj_vals * \
			(1 - eta_ind) + c_prime_noadj_vals * eta_ind

		c_prime_val_norent[c_prime_val_norent <= C_min] = C_min
		c_prime_val_norent[np.isnan(c_prime_val_norent)] = C_min

		h_prime_val_norent = h_prime_adj_vals * \
			(1 - eta_ind) + h_prime_noadj_vals * eta_ind

		h_prime_val_norent[h_prime_val_norent <= H_min] = H_min
		h_prime_val_norent[np.isnan(h_prime_val_norent)] = H_min


		# t+2 states if not renting and discounred t+2 utility (on t+1
		# information)

		# t+2 period utilities and marginal utilities conditioned on end of t+1
		# period mortgage leverage at t+1 prices
		UF_dp_val_norent = UF_dp_val_noadj*eta_ind + (1-eta_ind)*UF_dp_val_adj

		#   t+1 marginal utility of consumption for non-retning

		uc_prime_norent = uc_vec(c_prime_val_norent,
							 h_prime_val_norent, alpha_housing)
		# STEP 9: combine all  renter policies

		h_prime_rent_val = eval_linear_c(W_Q_R, h_prime_rent,
									   state_prime_rent, xto.NEAREST)

		#c_prime_rent_val = phi_r * Q_prime * h_prime_rent_val\
		#	* (1 - alpha_housing) / alpha_housing

		c_prime_rent_val = ch_ser(h_prime_rent_val, alpha_housing, phi_r*Q_prime)

		c_prime_rent_val[c_prime_rent_val <= C_min] = C_min
		c_prime_rent_val[np.isnan(c_prime_rent_val)] = C_min
		h_prime_rent_val[h_prime_rent_val <= H_min] = H_min
		h_prime_rent_val[np.isnan(h_prime_rent_val)] = H_min

		a_prime_rent_val = state_prime_rent[:, 1] - c_prime_rent_val\
			- h_prime_rent_val * phi_r * Q_prime

		a_prime_rent_val[a_prime_rent_val <= A_min] = A_min

		state_dp_prime_rent = np.column_stack((a_prime_rent_val, np.full(
				len(a_prime_rent_val), H_min), Q_prime, np.full(
				len(a_prime_rent_val), 0)))

		#  t+1 marginal utility with renting
		uc_prime_rent = uc_vec(c_prime_rent_val,
						   h_prime_rent_val, alpha_housing)

		u_prime_rent = u_vec(c_prime_rent_val,
						 h_prime_rent_val, alpha_housing)

		u_prime_norent = u_vec(c_prime_val_norent,
						   h_prime_val_norent, alpha_housing)

		UF_dp_val_rent = beta_bar * \
			eval_linear_c(X_cont_R, UF_dbprime, state_dp_prime_rent, xto.NEAREST)

		# STEP 10: make renting vs. no renting decision  and combine all
		# policies

		renter = u_prime_rent + UF_dp_val_rent > u_prime_norent + UF_dp_val_norent

		VF_db_prime = renter * UF_dp_val_rent + (1-renter)*UF_dp_val_norent
		h_prime_val = renter * h_prime_rent_val + \
			(1 - renter) * h_prime_val_norent
		c_prime_val = renter * c_prime_rent_val + \
			(1 - renter) * c_prime_val_norent
		uc_prime = renter * uc_prime_rent + (1 - renter) * uc_prime_norent

		# STEP 11: t+1 utilities conditioned in t info (renter)

		UC_prime = np.dot(s[t] * uc_prime * (1 + r) + (1-s[t])*(1+r)*b_prime(A_prime),
						Q_DC_P)

		UC_prime_H_rent = s[t]*Q_prime*uc_prime_rent + (1-s[t])*Q_prime*b_prime(A_prime)
		UC_prime_H_adj = s[t]*Q_prime*uc_prime_norent + (1-s[t])*Q_prime*b_prime(A_prime)
		UC_prime_H_noadj = s[t]*UH_dp_val_norent + (1-s[t])*Q_prime*b_prime(A_prime)
		UC_prime_H_inner = renter*UC_prime_H_rent\
							+ (1-renter)*eta_ind*UC_prime_H_noadj\
							+ (1-renter)*(1-eta_ind)*UC_prime_H_adj

		UC_prime_H = np.dot(Q_DC_P,UC_prime_H_inner )

		# Below is derivative of t+1 utility wrt to t+1 mortgage liability 
		# before interest and at t prices
		UC_prime_M_inner = s[t] * M_prime_R * uc_prime \
							 	+ (1 - s[t]) * M_prime_R * b_prime(A_prime)  

		UC_prime_M = np.dot(UC_prime_M_inner, Q_DC_P)
		c_prime_val[c_prime_val <= 0] = C_min
		c_prime_val[np.isnan(c_prime_val)] = C_min
		h_prime_val[h_prime_val <= 0] = H_min
		h_prime_val[np.isnan(h_prime_val)] = H_min
		
		UF_inner = u_vec(c_prime_val, h_prime_val, alpha_housing) +  VF_db_prime

		UF = np.dot(s[t] * UF_inner + (1 - s[t]) * b(A_prime), Q_DC_P)

		# discount everything back
		UC_prime_RHS = beta_bar * UC_prime
		UC_prime_H_RHS = beta_bar * UC_prime_H
		UC_prime_M_RHS = beta_bar * UC_prime_M

		return UC_prime_RHS, UC_prime_H_RHS,UC_prime_M_RHS, UF

	@njit
	def eval_mort_policy(t, UC_prime, UC_prime_M):
		""" returns unconstrained next period mortgage m_t+1 as function
			of a_t+1, h_t and q_t

			Note
			----
			- We calc m_t+1 in terms of t house prices

			"""
		m_prime_func = np.empty(grid_size_A * grid_size_H
								* grid_size_Q)

		# loop over values of A_t+1, H_t, Q_t
		for i in range(len(X_RC_contgp)):

			# pull out state values for i
			x_prime, h, q = X_RC_contgp[i][0], X_RC_contgp[i][1],\
				X_RC_contgp[i][2]
			m_mort_args = (x_prime, h, q, UC_prime,UC_prime_M, t)
			points_max = np.array([x_prime, h,q, M[-1]])
			points_full_pay = np.array([x_prime, h,q,0])

			# get RHS of Euler equation when max. mortgage taken
			# (max given by grid max)
			UC_prime_RHSm = eval_linear_c(X_cont_R, UC_prime, points_max, xto.NEAREST)
			UC_prime_M_RHSm = eval_linear_c(X_cont_R, UC_prime_M, points_max, xto.NEAREST)

			# get RHS of Euler when min mortage taken
			# (no negative mortgages )
			UC_prime_RHSf = eval_linear_c(X_cont_R, UC_prime, points_full_pay, xto.NEAREST)
			UC_prime_M_RHSf = eval_linear_c(X_cont_R, UC_prime_M, points_full_pay, xto.NEAREST)

			if mort_FOC(0, *m_mort_args)\
					* mort_FOC(M[-1], *m_mort_args) < 0:
				m_prime_func[i] = brentq(mort_FOC, 0, M[-1],
										 args=m_mort_args, disp = False)[0]
				if m_prime_func[i] == np.nan:
					m_prime_func[i] = 0

			# check if m_t+1 is constrained by max mortgage
			elif UC_prime_RHSm > UC_prime_M_RHSm:
				m_prime_func[i] = M[-1]

			# check if m_t+1 is constrained by min mortgage
			elif UC_prime_RHSf < UC_prime_M_RHSf:
				m_prime_func[i] = 0

			# otherwise, put in a zero mort_FOC
			else:
				m_prime_func[i] = 0
		# Reshape to wide and return function
		return m_prime_func.reshape(grid_size_A, grid_size_H, grid_size_Q)

	@njit
	def eval_policy_R_noadj(t, m_prime_func, UC_prime):
		"""Generates time t policy functions for housing non-adjusters

		Parameters
		----------
		t : int
			 age
		t_prime_funcs :     6-tuple
							 t+1 policy functions

		Returns
		-------
		a_prime_noadj:  4D array
						 t liquid asset function no-adj
						 defined on t+1 AxHxQxM

		c_prime_noadj:  4D array
						 t consumption function no-adj
						 defined on t+1 AxHxQxM

		eta_prime_noadj:4D array
						 t eta function no-adj
						 defined on t+1 AxHxQxM

		Note: Age t A is assets before returns, H is housing after dep.
		and W is wealth in hand after returns and sale of house at
		with current period prices

		"""
		# generate endogenous grid and eta_h_t

		assets_endgrid_1 = np.empty(grid_size_A * grid_size_H
									* grid_size_Q * grid_size_M)
		cons_1 = np.empty(grid_size_A * grid_size_H
						  * grid_size_Q * grid_size_M)

		# loop over values of A_t+1, H_t, Q_t, M_t
		for i in range(len(X_R_contgp)):

			x_cont_vals = X_R_contgp[i]
			#print(x_cont_vals)

			ap_ind, h_ind, q_ind = X_R_cont_ind[i][0],\
				X_R_cont_ind[i][1],\
				X_R_cont_ind[i][2]

			# return optimal next period mortgage value and
			# t period consumption, asssign consumption to grid

			cons_1[i], m_prime = eval_c_mort(x_cont_vals[0],
											 x_cont_vals[1],
											 x_cont_vals[2],
											 m_prime_func[:, h_ind,
														  q_ind],
											 UC_prime, t)
		   #print(m_prime)

			# calculate extra mortgage payment i.e. pay above min. amort
			extra_payment_made = (x_cont_vals[3] - m_prime) * x_cont_vals[1] * x_cont_vals[2]
			#print(extra_payment_made)

			# assign A_t value to endogenous grid

			assets_endgrid_1[i] = cons_1[i] + x_cont_vals[0]\
				+ extra_payment_made

			# eval. RHS values of Euler at optimum
			#UC_prime_RHS, UC_prime_H_RHS,\
			#	UC_prime_M_RHS, UF\
			#	= gen_UC_RHS(t, x_cont_vals[0],
			#				 x_cont_vals[1],
			#				 x_cont_vals[2],
			#				 m_prime,
			#				 *t_prime_funcs)

		# interpolate A_t+1, C_t and eta_t on endogenous
		# grid points, i.e on time t states

		a_noadj, c_noadj = interp_no_adj(assets_endgrid_1, cons_1)

		return a_noadj, c_noadj
	@njit
	def eval_policy_R_adj(t, m_prime_func, U_prime_funcs):
		""" Generate policy functions with housing stcok adjustment
				and non-zero liquid saving A_t+1
		"""

		a_adj = np.zeros(grid_size_H * grid_size_Q )
		wealth_endgrid = np.zeros(grid_size_H * grid_size_Q)
		c_adj = np.zeros(grid_size_H * grid_size_Q )
		(UF_prime, UC_prime, UC_prime_H, UC_prime_M) = U_prime_funcs

		for i in range(len(X_H_R_ind)):
			h_index = X_H_R_ind[i][0]
			q_index = X_H_R_ind[i][1]
			#m_index = X_H_R_ind[i][2]


			args_HA_FOC = (H[h_index],
						   Q[q_index],
						   m_prime_func[:, h_index, q_index],
						   UF_prime, UC_prime, UC_prime_H, UC_prime_M, t)

			args_H_FOC = (A_min,H[h_index],\
							Q[q_index],\
							UF_prime, UC_prime, UC_prime_H, UC_prime_M,t)

			# Check if interior solution for a_t+1 exists

			

			if int(h_index) == 0:
				a_adj[i] = A_min

				c_adj[i] = C_min / 2
				m_prime1 = min(max(HA_FOC(a_adj[i],
										  H[h_index],
										  Q[q_index],
										  m_prime_func[:,h_index, q_index],
										  UF_prime,
										  UC_prime,
										  UC_prime_H,
										  UC_prime_M,
										  t,
										  ret_mort=True), 0),M[-1])

				wealth_endgrid[i] = c_adj[i] + a_adj[i] + Q[q_index]\
					* H[h_index]\
					* (1 - m_prime1 + tau_housing)
				#print(wealth_endgrid[i])

			elif HA_FOC(A_min, *args_HA_FOC) * HA_FOC(100, *args_HA_FOC) < 0:

				# If interior solution to a_t+1, calculate it
				a_adj[i] = max(brentq(HA_FOC, A_min, 100,
									  args=args_HA_FOC, disp = False)[0], A_min)

				c_adj[i] = max(HA_FOC(a_adj[i], H[h_index],
									  Q[q_index],
									  m_prime_func[:, h_index, q_index],
									  UF_prime,
										  UC_prime,
										  UC_prime_H,
										  UC_prime_M,t, ret_cons=True), C_min)

				m_prime1 = min(max(HA_FOC(a_adj[i],
										  H[h_index],
										  Q[q_index],
										  m_prime_func[:,h_index, q_index],
										  UF_prime,
										  UC_prime,
										  UC_prime_H,
										  UC_prime_M,
										  t,
										  ret_mort=True), 0),M[-1])

				wealth_endgrid[i] = c_adj[i] + a_adj[i] + Q[q_index]\
					* H[h_index]\
					* (1 - m_prime1 + tau_housing)\
					
			elif H_FOC(C_min, *args_H_FOC)*H_FOC(C_max, *args_H_FOC)<0:
				a_adj[i] = A_min 
				c_adj[i] = max(brentq(H_FOC, C_min,C_max,\
								args = args_H_FOC, disp = False)[0], C_min)


				m_prime1 = min(max(H_FOC(c_adj[i], a_adj[i],
										  H[h_index],
										  Q[q_index],
										  UF_prime,
										  UC_prime,
										  UC_prime_H,
										  UC_prime_M,
										  t,
										  ret_mort=True), 0),M[-1])

				wealth_endgrid[i] = c_adj[i] + a_adj[i] + Q[q_index]\
					* H[h_index]\
					* (1 - m_prime1 + tau_housing)\
					

			else:
				a_adj[i] = np.nan
				c_adj[i] = np.nan
				wealth_endgrid[i] = np.nan

		a_adj_uniform, c_adj_uniform, H_adj_uniform \
			= interp_adj(a_adj, c_adj, wealth_endgrid, extrap=True)

		return a_adj_uniform, c_adj_uniform, H_adj_uniform

	@njit
	def gen_uf_prime(t, t_prime_funcs):

		uf_prime_1 = np.zeros(len(X_R_contgp))
		uc_prime_1 = np.zeros(len(X_R_contgp))
		uh_prime_1 = np.zeros(len(X_R_contgp))
		um_prime_1 = np.zeros(len(X_R_contgp))

		for i in range(len(X_R_contgp)):
			x_cont_vals = X_R_contgp[i]

			UC_prime_RHS, UC_prime_H_RHS,\
				UC_prime_M_RHS, UF\
				= gen_UC_RHS(t, x_cont_vals[0],
								x_cont_vals[1],\
								x_cont_vals[2],\
								x_cont_vals[3],
								*t_prime_funcs)

			uf_prime_1[i] = UF
			uh_prime_1[i] = UC_prime_H_RHS
			um_prime_1[i] = UC_prime_M_RHS
			uc_prime_1[i] = UC_prime_RHS


		uf_prime_2 = uf_prime_1.reshape((grid_size_A,\
									 grid_size_H,\
									 grid_size_Q,\
									 grid_size_M))
		uh_prime_1 = uh_prime_1.reshape((grid_size_A,\
									 grid_size_H,\
									 grid_size_Q,\
									 grid_size_M))
		uc_prime_1 = uc_prime_1.reshape((grid_size_A,\
					 grid_size_H,\
					 grid_size_Q,\
					 grid_size_M))
		um_prime_1 = um_prime_1.reshape((grid_size_A,\
					 grid_size_H,\
					 grid_size_Q,\
					 grid_size_M))

		return uf_prime_2, uc_prime_1,uh_prime_1, um_prime_1 

	@njit
	def gen_rhs_val_adj(t, points,
						a_prime_adj,
						c_prime_adj,
						h_prime_adj, VF):
		""" Retrun value of interpolated  policy
			functions for housing adjuster at
			points"""

		# policies with liquid saving

		H_prime_adj_val = eval_linear_c(X_QH_R,
									  h_prime_adj,
									  points,
									  xto.NEAREST)

		H_prime_adj_val[H_prime_adj_val < H_min] = H_min

		c_prime_adj_val = eval_linear_c(X_QH_R,
									  c_prime_adj,
									  points, xto.NEAREST)

		c_prime_adj_val[c_prime_adj_val < C_min] = C_min

		a_prime_adj_val = eval_linear_c(X_QH_R,
									  a_prime_adj,
									  points,
									  xto.NEAREST)

		vf_val = eval_linear_c(X_QH_R,
									  a_prime_adj,
									  points,
									  xto.NEAREST) 

		a_prime_adj_val[a_prime_adj_val < A_min] = A_min

		end_of_period_liability = points[:, 2] - c_prime_adj_val\
			- H_prime_adj_val* (1 + tau_housing)* points[:, 1]\
			- a_prime_adj_val

		#end_of_period_liability = points[:,3] * points[:,0] * points[:,1]\
		#							 - extra_pay_adj_val

		mort_dp_prime = end_of_period_liability\
							 / (H_prime_adj_val * points[:, 0])

		mort_dp_prime[mort_dp_prime <= 0] = 0
		mort_dp_prime[mort_dp_prime > M[-1]] = 1

		H_prime_val = H_prime_adj_val
		c_prime_val = c_prime_adj_val
		a_prime_val = a_prime_adj_val

		return c_prime_val, H_prime_val, a_prime_val,\
			mort_dp_prime, vf_val

	@njit
	def gen_rhs_val_noadj(t, points,
						  a_prime_noadj,
						  c_prime_noadj, VF, UH_prime):
		""" Interpolate value of interped policy
			functions for housing non-adjuster at
			points"""

		#etavals = eval_linear_c(X_cont_R,
		#					  eta_prime_noadj,
		#					  points,
		#					  xto.NEAREST)

		H_prime_noadj_val = points[:, 1] * (1 - delta_housing)

		c_prime_noadj_val = eval_linear_c(X_cont_R,
										c_prime_noadj,
										points,
										xto.NEAREST)

		c_prime_noadj_val[c_prime_noadj_val < C_min] = C_min

		a_prime_noadj_val = eval_linear_c(X_cont_R,
										a_prime_noadj,
										points,
										xto.NEAREST)

		a_prime_noadj_val[a_prime_noadj_val < 0] = 0

		extra_pay_noadj_val = points[:, 0] - a_prime_noadj_val\
			- c_prime_noadj_val

		m_dp_prime = - (extra_pay_noadj_val /
						points[:, 0] * points[:, 1]) + points[:, 3]

		mort_db_prime_noadj = m_dp_prime

		mort_db_prime_noadj[mort_db_prime_noadj < 0] = 0
		mort_db_prime_noadj[mort_db_prime_noadj > 1] = 1

		vf_val = eval_linear_c(X_cont_R,
							VF,
							points,
							xto.NEAREST)
		uh_prime_val = eval_linear_c(X_cont_R,
							UH_prime,
							points,
							xto.NEAREST)

		return c_prime_noadj_val, H_prime_noadj_val,\
				 mort_db_prime_noadj, vf_val, uh_prime_val

	@njit
	def gen_rhs_val_rent(t, points,
						 h_prime_rent):
		""" Interpolate value of interped policy
			functions for housing non-adjuster at
			points"""

		h_prime_rent_val = eval_linear_c(W_Q_R,
									   h_prime_rent,
									   points,
									   xto.NEAREST)

		c_prime_rent_val = phi_r * points[0, 1] * h_prime_rent_val\
			* (1 - alpha_housing) / alpha_housing

		c_prime_rent_val[c_prime_rent_val <= C_min] = C_min
		h_prime_rent_val[h_prime_rent_val <= H_min] = H_min

		return c_prime_rent_val, h_prime_rent_val


	def gen_RHS_TR(comm, t, a_prime_noadj,
					c_prime_noadj,
					a_prime_adj,
					c_prime_adj,
					h_prime_adj,
					h_prime_rent,
					UF_dbprime, UH_prime):

		"""Generate RHS time T_R Euler equation conditioned on:
			- housing stock taken into time T_R (H_{TR-1})
			- DC assets (before returns) taken into into time T_R
			- mortage liability (before interest) taken into time T_R
			- T_R-1 housing stock
			- liquid assets taken into time T_R (before returns)
			- T_R -1 wage shock, alpha, beta shock, Pi
			- T_R- 1 house price
			- DB/DC

			First index of output corresponds to discrete index in cart
			prod of disctete exog states

		Parameters
		----------
		t :                         int
									 age
		assets_prime_uniform:     2D array
									 no adjust a_t+1 on t cont. cart
									 grid
		etas_prime_uniform:       2D array
									 eta_t defined on t continuous cart

		H_prime_adj:              3D array
									 adj. H_t on Q_t x W_t
		assets_prime_uniform_adj: 3D array
									 adj a_t+1 on Q_t x W_ts

		Returns
		-------
		UC_prime_out:               10D array

		UC_prime_H_out:             10D array

		UC_prime_HFC_out:           10D array

		UC_prime_M_out:             10D array

		Lamba:                      10D array

		VF:                         10D array

		Notes
		-----
		Time T_R is first retirement year and all pension accounts are paid 
		at this age. 

		"""

		def my_gen_RHS_TR(my_X_all_hat_ind, DB_payout):
			"""
			Generates RHS retiree Euler for each of the (my) worker grid points. 
			Loop over states, where each i indexes a cartesian product of:

			0 - DB/DC
			1 - E     (TR-1, previous period)
			2 - alpha (TR-1, previous period)
			3 - beta  (TR-1, previous period)
			4 - Pi    (TR-1, previous period)
			5 - A *before returns* at T_R
			6 - A_DC *before returns* taken into T_R
			7 - H at T_R (coming into state,
				 before T_R depreciation)
			8 - Q at T_R (previous period)
			9 - M at T_R (coming into state,
				  before T_R interest)

			"""

			# Make empty arrays to fill with RHS Euler values 
			my_UC_prime_out = np.zeros(len(my_X_all_hat_ind))
			my_UC_prime_H_out = np.zeros(len(my_X_all_hat_ind))
			my_UC_prime_M_out = np.zeros(len(my_X_all_hat_ind))
			my_Lambda_out = np.zeros(len(my_X_all_hat_ind))
			my_VF = np.zeros(len(my_X_all_hat_ind))

			@njit 
			def _my_gen_RHS_TR_point(i):
				""" Function generates RHS for grid point 
					i by looping over next period wage shocks
					and summing over markov probs"""

				# Pull out the i state values 	
				q_in = Q[my_X_all_hat_ind[i][8]]
				H_in = H[my_X_all_hat_ind[i][7]]
				q_ind = my_X_all_hat_ind[i][8]
				E_ind = my_X_all_hat_ind[i][1]
				ADC_in = A_DC[my_X_all_hat_ind[i][6]]
				r_share = Pi[my_X_all_hat_ind[i][4]]
				m_in = M[my_X_all_hat_ind[i][9]]

				# Generate values for relisations of T_R period
				# house price shocks, DC values after returns
				# mortgage interest shocks and mortgage balances
				# after interest.
				Q_prime = q_in * (1 + r_H + Q_DC_shocks[:, 2])
				A_DC_prime = ((1 - r_share) * Q_DC_shocks[:, 0]
							  + r_share * Q_DC_shocks[:, 1]) * ADC_in
				q_t_arr = np.full(len(Q_DC_shocks[:, 2]), q_in)

				# t+1 leverage wrt. to t+1 house price after t+1 interest
				M_prime = r_m_prime*m_in*q_t_arr/(1-delta_housing) * Q_prime 
				M_prime_R = r_m_prime 

				UC_prime = np.zeros(len(M_prime)) 
				UC_prime_H = np.zeros(len(M_prime)) 								  
				UC_prime_M = np.zeros(len(M_prime)) 
				VF_cont = np.zeros(len(M_prime)) 
				
				# For each T_R-1 period exogenous state,
				# loop over R period wage stock realisation and sum 
				for j in range(len(E)):

					# liquid wealth before DC pay out
					# but including DB pay out
					a_l_exDC = DB_payout[j]\
						* (1 - acc_ind)\
						+ (1 + r) * A[my_X_all_hat_ind[i][5]]

					# Add DC pay-out to wealth 
					a_l = A_DC_prime + a_l_exDC

					# Wealth net of mortgage liability for adjusters
					wealth = a_l + Q_prime * H_in * \
						(1 - delta_housing) * (1 - M_prime)

					#A_prime is bequethed wealth if agent dies at the beggining 
					# of period.
					A_prime = wealth
					A_prime[A_prime <= 0] = A_min

					# Turn housing at beggining of R period into array
					h_prime_arr = np.full(len(Q_prime),
										  (1 - delta_housing) * H_in)

					# State points of length housing price shock x DC shock
					# interpolation policy functions over these points.
					point_noadj = np.column_stack((a_l, h_prime_arr,\
													 Q_prime, M_prime))
					points_adj = np.column_stack((Q_prime, wealth))
					points_rent = np.column_stack((Q_prime, wealth))

					# Generate the policy functions for adjuster,
					# non-adjuster and renter 
					c_prime_val_noadj, H_prime_val_noadj,\
						mort_db_prime_noadj, vf_val_noadj, uh_db_prime =\
						gen_rhs_val_noadj(t, point_noadj,
										  a_prime_noadj,
										  c_prime_noadj,
										  UF_dbprime, UH_prime)

					c_prime_val_adj, H_prime_val_adj, a_prime_val_adj,\
						mort_dp_prime_adj, vf_val_adj =\
						gen_rhs_val_adj(t, points_adj,
										a_prime_adj,
										c_prime_adj,
										h_prime_adj,
										UF_dbprime)

					c_prime_val_rent, h_prime_val_rent = \
						gen_rhs_val_rent(t, points_rent,
										 h_prime_rent)

					# Now the value functions at R
					val_noadj = u_vec(c_prime_val_noadj, H_prime_val_noadj,\
									 alpha_housing) + beta_bar * vf_val_noadj
					val_adj = u_vec(c_prime_val_adj, H_prime_val_adj, alpha_housing)\
								+ beta_bar * vf_val_adj
					eta_ind = val_noadj > val_adj
					val_norent = eta_ind*val_noadj + (1-eta_ind)*val_adj
								
					# Combine non-renting policies
					c_prime_val = eta_ind * c_prime_val_noadj\
						+ (1 - eta_ind) * c_prime_val_adj
					H_prime_val = eta_ind * H_prime_val_noadj\
						+ (1 - eta_ind) * H_prime_val_adj
					a_prime_val = eta_ind * H_prime_val_noadj\
						+ (1 - eta_ind) * a_prime_val_adj
					mort_db_prime = eta_ind * mort_db_prime_noadj\
						+ (1 - eta_ind) * mort_dp_prime_adj

					# Non-renter marginal utilities 
					uc_prime_norent = uc_vec(c_prime_val, H_prime_val,
										 	alpha_housing)

					u_norent = u_vec(c_prime_val, H_prime_val,
								 			alpha_housing)

					# Policies with renting
					uc_prime_rent = uc_vec(c_prime_val_rent,
									   h_prime_val_rent,
									   alpha_housing)

					u_rent = u_vec(c_prime_val_rent,
							   c_prime_val_rent,
							   alpha_housing)

					a_prime_rent_val = points_rent[:, 1] - c_prime_val_rent\
						- h_prime_val_rent\
						* phi_r * Q_prime

					a_prime_rent_val[a_prime_rent_val <= A_min] = A_min

					state_dp_prime_rent = np.column_stack(
						(a_prime_rent_val, np.full(
							len(a_prime_rent_val), H_min), Q_prime, np.full(
							len(a_prime_rent_val), 0)))

					UF_dp_val_rent = beta_bar * eval_linear_c(
						X_cont_R, UF_dbprime, state_dp_prime_rent, xto.NEAREST)
					val_rent = u_rent + UF_dp_val_rent
				   
					# Index to rent or not.
					rent_ind = val_rent > val_norent

					# Combine R period marginal utilities and value function 
					# across renters and non-renters
					uc_prime = rent_ind * uc_prime_rent \
						+ (1 - rent_ind) * uc_prime_norent
					val = rent_ind * val_rent \
						+ (1 - rent_ind) * val_norent

					# Generate combined marginal utilities
					# wrt liq. assets, housing, adjusting housing, mortages
					# DC assets and R period utility value
					#
					# we have multiplied functions with probability of wage shock for state j
					# conditioned on state E_ind in the previous period
					# *note we sum over the j wage shock probs in the loop over len(Q_DC_P)*

					UC_prime = UC_prime+  P_E[E_ind][j]\
								 * ((1 + r) * (s[int(R - 1)] * uc_prime \
									+ (1 - s[int(R - 1)]) * b_prime(A_prime)))

					UC_prime_H_noadj = beta_bar * (1 - delta_housing) * uh_db_prime
					UC_prime_H_adj_rent = Q_prime * (1 - delta_housing) * uc_prime
					index_adj_or_rent = (1-eta_ind) * (1-rent_ind) + rent_ind
					index_no_adj_and_norent = eta_ind * (1-rent_ind)
					m_beq_h_all  = Q_prime * (1 - delta_housing) * b_prime(A_prime)
					
					UC_prime_H = UC_prime_H +  P_E[E_ind][j] * (s[int(R - 1)]*(\
										(index_adj_or_rent * UC_prime_H_adj_rent \
											+ index_no_adj_and_norent * UC_prime_H_noadj))\
											+ (1 - s[int(R - 1)])*m_beq_h_all)

					UC_prime_M = UC_prime_M+  P_E[E_ind][j] * M_prime_R * (s[int(R - 1)] * uc_prime\
											 + (1 - s[int(R - 1)]) * b_prime(A_prime)) # check the bequest here, interest is not being charged 

					VF_cont = VF_cont + s[int(R - 1)] * P_E[E_ind][j] * val +\
									P_E[E_ind][j] *(1 - s[int(R - 1)]) * b(A_prime)

				# No condition over the i.i.d shocks
				my_UC_prime_out_i = np.dot(Q_DC_P,  UC_prime)
				my_UC_prime_H_out_i = np.dot(Q_DC_P,  UC_prime_H)
				my_UC_prime_M_out_i = np.dot(Q_DC_P,  UC_prime_M)
				my_VF_i = np.dot(Q_DC_P,  VF_cont)

				return my_UC_prime_out_i, my_UC_prime_H_out_i,\
						 my_UC_prime_M_out_i, my_VF_i

			for i in range(len(my_X_all_hat_ind)):
				my_UC_prime_out[i], my_UC_prime_H_out[i],\
				my_UC_prime_M_out[i],my_VF[i] = _my_gen_RHS_TR_point(i)


			return my_UC_prime_out, my_UC_prime_H_out,\
					 my_UC_prime_M_out, my_VF

		DB_payout = None 
		
		if comm.rank ==0:

			X_all_hat_ind = X_all_hat_ind_func().astype(np.int32)
			X_all_hat_ind_split = np.array_split(X_all_hat_ind,\
												 comm.size, axis = 0)
			UC_prime_out = np.zeros(len(X_all_hat_ind))
			UC_prime_H_out = np.zeros(len(X_all_hat_ind))
			UC_prime_M_out = np.zeros(len(X_all_hat_ind))
			Lambda_out = np.zeros(len(X_all_hat_ind))
			VF = np.zeros(len(X_all_hat_ind))
			del X_all_hat_ind
			gc.collect()

			# array of possible DB pay-outs for this age
			DB_payout = np.zeros(len(E))
			for i in range(len(E)):
				DB_payout[i] = DB_benefit(t, t - tzero,
										  y(t, E[i]),
										  i,
										  P_E,
										  P_stat,
										  E)

		else:

			X_all_hat_ind_split = None
			UC_prime_out = None
			UC_prime_H_out = None
			UC_prime_M_out = None
			VF = None
			X_all_hat_ind = None

		DB_payout = comm.bcast(DB_payout, root = 0)
		my_X_all_hat_ind = comm.scatter(X_all_hat_ind_split, root = 0)
		my_UC_prime_out, my_UC_prime_H_out, my_UC_prime_M_out, my_VF\
			= my_gen_RHS_TR(my_X_all_hat_ind, DB_payout)
		del X_all_hat_ind_split

		sendcounts = np.array(comm.gather(len(my_UC_prime_out), 0)) 

		comm.Gatherv(np.ascontiguousarray(my_UC_prime_out),\
					 recvbuf=(UC_prime_out, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_UC_prime_H_out),\
					 recvbuf=(UC_prime_H_out, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_UC_prime_M_out),\
					recvbuf=(UC_prime_M_out, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_VF),\
					recvbuf=(VF, sendcounts), root =0)

		del my_UC_prime_out, my_UC_prime_H_out,my_UC_prime_M_out, my_VF
		del my_X_all_hat_ind
		gc.collect()

		if comm.rank == 0:
			return UC_prime_out, UC_prime_H_out, UC_prime_M_out, VF
		else:
			return None, None, None, None

	def gen_R_pol(comm, noplot= True):

		a_noadj = np.zeros((grid_size_A, grid_size_H, grid_size_Q, grid_size_M))
		c_noadj = np.zeros((grid_size_A, grid_size_H, grid_size_Q, grid_size_M))
		a_adj_uniform = np.zeros((grid_size_Q, grid_size_A))
		c_adj_uniform = np.zeros((grid_size_Q, grid_size_A))
		H_adj_uniform = np.zeros((grid_size_Q, grid_size_A))
		h_prime_rent = np.zeros((grid_size_A, grid_size_Q))
		m_prime_func = np.empty((grid_size_A, grid_size_H, grid_size_Q))
		UF_prime = np.zeros((grid_size_A, grid_size_H, grid_size_Q, grid_size_M))
		UH_prime = np.zeros((grid_size_A, grid_size_H, grid_size_Q, grid_size_M))

		t_prime_funcs = (a_noadj, c_noadj, a_adj_uniform, c_adj_uniform,
					 H_adj_uniform, h_prime_rent, UF_prime, UH_prime)

		UF_prime, UC_prime, UC_prime_H, UC_prime_M = gen_uf_prime(T, t_prime_funcs)

		t_prime_funcs = (a_noadj, c_noadj, a_adj_uniform, c_adj_uniform,
					 		H_adj_uniform, h_prime_rent, UF_prime, UC_prime_H)

		U_prime_funcs = (UF_prime, UC_prime, UC_prime_H, UC_prime_M)

		for i in range(int(T - R + 1)):
			t = T - i
			
			start = time.time()
			if comm.rank == 0:
				#print(t)
				h_prime_rent = eval_rent_pol(UC_prime, t)

			elif comm.rank == 1:
				m_prime_func = eval_mort_policy(t, UC_prime,UC_prime_M)
				comm.Send(m_prime_func, dest = 0, tag= 11)

			if comm.rank == 0:
				comm.Recv(m_prime_func, source = 1, tag = 11)
				a_noadj, c_noadj = eval_policy_R_noadj(
					t, m_prime_func, UC_prime)

			elif comm.rank == 1:
				a_adj_uniform, c_adj_uniform, H_adj_uniform = eval_policy_R_adj(
					t, m_prime_func, U_prime_funcs)
				comm.Send(a_adj_uniform, dest = 0, tag = 21)
				comm.Send(c_adj_uniform, dest = 0, tag = 23)
				comm.Send(H_adj_uniform, dest = 0, tag = 24)

			else:
				pass 

			if comm.rank ==0:
				comm.Recv(a_adj_uniform, source = 1, tag = 21)
				comm.Recv(c_adj_uniform ,source = 1, tag = 23)
				comm.Recv(H_adj_uniform, source = 1, tag = 24)

				t_prime_funcs =\
					(a_noadj, c_noadj, a_adj_uniform, c_adj_uniform,
					 H_adj_uniform, h_prime_rent, UF_prime, UC_prime_H)

				UF_prime, UC_prime, UC_prime_H, UC_prime_M\
					 = gen_uf_prime(T, t_prime_funcs)
			else:
				pass 

			UF_prime = comm.bcast(UF_prime, root=0)
			UC_prime = comm.bcast(UC_prime, root=0)
			UC_prime_H = comm.bcast(UC_prime_H, root=0)
			UC_prime_M = comm.bcast(UC_prime_M, root=0)

			U_prime_funcs = (UF_prime, UC_prime, UC_prime_H, UC_prime_M)

		start = time.time()
		t_prime_funcs = comm.bcast(t_prime_funcs, root = 0)
		UC_prime_out, UC_prime_H_out, UC_prime_M_out, VF\
			 = gen_RHS_TR(comm, t, *t_prime_funcs)

		if comm.rank == 0:
			if noplot == False:
				return  a_noadj, c_noadj, a_adj_uniform, c_adj_uniform, H_adj_uniform,\
						h_prime_rent, UC_prime_out, UC_prime_H_out, UC_prime_M_out, VF
			if noplot == True:
				return  UC_prime_out, UC_prime_H_out, UC_prime_M_out, VF
		else:
			if noplot == True:
				return None, None,None, None
			else:
				return None, None,None, None, None,\
						None,None, None, None, None

	return gen_R_pol

if __name__ == "__main__":
	pass
	