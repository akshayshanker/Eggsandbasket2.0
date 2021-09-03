"""Module generates operator to solve worker policy functions

Functions: worker_solver_factory
			Generates  operator solve an instance of LifeCycleModel

Example use:

gen_R_pol = retiree_func_factory(lifecycle_model)

solve_LC_model = worker_solver_factory(lifecycle_model,gen_R_pol)

policy = solve_LC_model = worker_solver_factory(og, world_comm, comm, gen_R_pol,
											scr_path = scr_path,
											gen_newpoints = gen_newpoints,
											verbose = False)
"""

# Import packages
import sys
from psutil import virtual_memory


sys.path.append("..")
from eggsandbaskets.util.helper_funcs import gen_policyout_arrays, d0, interp_as,\
	gen_reshape_funcs, einsum_row
from eggsandbaskets.util.grids_generate import generate_points
from pathlib import Path
from pympler import tracker

import dill as pickle
from interpolation import interp
from interpolation.splines import extrap_options as xto
from interpolation.splines import eval_linear
from quantecon.optimize.root_finding import brentq
from numba import njit
import time
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')
from mpi4py import MPI


def worker_solver_factory(og,
							world_comm,
							comm,
							gen_R_pol,
							scr_path,
							gen_newpoints = False,
							verbose = False):
	"""Generates operator that solves  worker policies

	Parameters
	----------
	og : LifeCycleModel
		  Instance of LifeCycleModel
	world_comm: MPI Communicator
				 Communicator for *all* CPUs across SMM estimation
	comm: MPI Communicator
			Communicator to solve this instance of LifeCycleModel
	gen_R_pol: Function
				function that generates retiring (T_R) age policy and RHS Euler
	scr_path: str
			   path to the scratch drive root
	gen_newpoints: Bool
					set to True if new pre-filled points generated and saved 
	Verbose: Bool

	Returns
	-------
	solve_LC_model: function
					 Operator that returns worker policies

	Note
	----
	gen_R_pol must be generated with the same instance of
	LifeCycleModel!

	Todo
	----
	Docstrings need to be completed for:
			- `eval_M_prime'
			-  `eval_pol_mort_W'
			- `gen_RHS_pol_noadj'

	More explaination of steps are required in:
			- `gen_RHS_pol_noadj'

	Check whether mortgage liability points in X_all_ind and in gen rhs
	func are before or after mortgage interst
	"""

	# Unpack  variables from class
	# functions
	mod_name = og.mod_name
	acc_ind = og.acc_ind
	ID = og.ID
	pol_path_id = scr_path + mod_name + '/' + ID + '_acc_' + str(acc_ind[0])
	Path(pol_path_id).mkdir(parents=True, exist_ok=True)
	pol_path_id_grid = scr_path + mod_name + '/grid'
	Path(pol_path_id_grid).mkdir(parents=True, exist_ok=True)

	u = og.functions.u
	uc = og.functions.ucnz
	uh = og.functions.uh
	uh_inv = og.functions.uh_inv

	u_vec, uc_vec = og.functions.u_vec, og.functions.uc_vec
	amort_rate = og.functions.amort_rate
	b, b_prime = og.functions.b, og.functions.b_prime
	y, yvec = og.functions.y, og.functions.yvec
	adj_p, adj_v, adj_pi = og.functions.adj_p, og.functions.adj_v,\
		og.functions.adj_pi
	housing_ser = og.functions.housing_ser
	ch_ser = og.functions.ch_ser
	ces_c1 = og.functions.ces_c1

	# parameters
	s = og.parameters.s
	delta_housing, tau_housing, def_pi = og.parameters.delta_housing,\
		og.parameters.tau_housing,\
		og.parameters.def_pi
	beta_bar, alpha_bar = og.parameters.beta_bar, og.parameters.alpha_bar

	# grid parameters
	DC_max = og.parameters.DC_max
	C_min, C_max = og.parameters.C_min, og.parameters.C_max
	Q_max, A_min = og.parameters.Q_max, og.parameters.A_min
	H_min, H_max = og.parameters.H_min, og.parameters.H_max
	A_max_W = og.parameters.A_max_W
	A_max_WW = og.parameters.A_max_WW
	T, tzero, R = og.parameters.T, og.parameters.tzero, og.parameters.R
	v_S, v_E = og.parameters.v_S, og.parameters.v_E
	r, r_l, r_H = og.parameters.r, og.parameters.r_l, og.parameters.r_H
	beta_m, kappa_m = og.parameters.beta_m, og.parameters.kappa_m
	beta_m, kappa_m = og.parameters.beta_m, og.parameters.kappa_m
	k = og.parameters.k
	phi_d = og.parameters.phi_d
	phi_c = og.parameters.phi_c
	phi_r = og.parameters.phi_r
	sigma_DC_V = og.parameters.sigma_DC_V
	sigma_DB_V = og.parameters.sigma_DB_V
	sigma_DC_pi = og.parameters.sigma_DC_pi
	sigma_DB_pi = og.parameters.sigma_DB_pi

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

	# exogenous shock processes
	beta_hat, P_beta, beta_stat = og.st_grid.beta_hat,\
		og.st_grid.P_beta, og.st_grid.beta_stat
	alpha_hat, P_alpha, alpha_stat = og.st_grid.alpha_hat,\
		og.st_grid.P_alpha, og.st_grid.alpha_stat

	X_r, P_r = og.st_grid.X_r, og.st_grid.P_r
	Q_shocks_r, Q_shocks_P = og.st_grid.Q_shocks_r,\
		og.st_grid.Q_shocks_P
	Q_DC_shocks, Q_DC_P = og.cart_grids.Q_DC_shocks,\
		og.cart_grids.Q_DC_P

	r_m_prime = og.cart_grids.r_m_prime

	EBA_P, EBA_P2 = og.cart_grids.EBA_P, og.cart_grids.EBA_P2

	E, P_E, P_stat = og.st_grid.E, og.st_grid.P_E, og.st_grid.P_stat

	# smaller, "fundamental" grids
	A_DC = og.grid1d.A_DC
	H = og.grid1d.H
	A = og.grid1d.A
	M = og.grid1d.M
	Q = og.grid1d.Q
	H_R = og.grid1d.H_R
	W_W = og.grid1d.W_W
	V, Pi, DB = og.grid1d.V, og.grid1d.Pi, og.grid1d.DB

	# medium sized and interpolation grids
	X_cont_W = og.interp_grid.X_cont_W
	X_cont_W_bar = og.interp_grid.X_cont_W_bar
	X_cont_W_hat = og.interp_grid.X_cont_W_hat
	X_DCQ_W = og.interp_grid.X_DCQ_W
	X_QH_W = og.interp_grid.X_QH_W
	X_cont_WAM = og.interp_grid.X_cont_WAM
	X_cont_WAM2 = og.interp_grid.X_cont_WAM2
	X_cont_AH = og.interp_grid.X_cont_AH
	X_W_contgp = og.interp_grid.X_W_contgp

	# larger grids
	if comm.rank == 0 or comm.rank == 2 or comm.rank == 3:
		X_all_ind = og.BigAssGrids.X_all_ind_f().astype(np.int32)
		X_all_hat_ind = og.BigAssGrids.X_all_hat_ind_f().astype(np.int32)
		X_all_B_ind = og.BigAssGrids.X_all_B_ind_f().astype(np.int32)
		x_all_ind_len_a = np.array([len(X_all_ind)])
		x_all_hat_ind_len = len(X_all_hat_ind)
		x_all_ind_len = x_all_ind_len_a[0]
		comm.Send(x_all_ind_len_a, dest=1, tag=111)
		comm.send(x_all_hat_ind_len, dest =1, tag= 1112)

	if comm.rank == 1:
		x_all_ind_len_a = np.array([0])
		x_all_hat_ind_len = None
		X_W_bar_hdjex_ind = og.BigAssGrids.X_W_bar_hdjex_ind_f().astype(np.int32)
		X_adj_func_ind = og.BigAssGrids.X_adj_func_ind_f().astype(np.int32)
		comm.Recv(x_all_ind_len_a, source=0, tag=111)
		comm.recv(x_all_hat_ind_len, source=0, tag=1112)
		x_all_ind_len = x_all_ind_len_a[0]

	X_all_C_ind = og.BigAssGrids.X_all_C_ind_f().astype(np.int32)
	X_V_func_DP_vals = og.BigAssGrids.X_V_func_DP_vals_f()
	X_V_func_CR_vals = og.BigAssGrids.X_V_func_CR_vals_f()

	# Preset shapes
	all_state_shape, all_state_shape_hat, v_func_shape,\
		all_state_A_last_shape, policy_shape_nadj,\
		policy_shape_adj, policy_shape_rent, prob_v_shape, prob_pi_shape = gen_policyout_arrays(og)

	start = time.time()
	# generate pre-filled interpolation points only on one node
	if gen_newpoints == True and world_comm.rank == 0:
		print("Generating interpolation points on world master.")
		
		points_gen_ind = generate_points(og, path = pol_path_id_grid)

	gc.collect()
	world_comm.barrier()
	if comm.rank == 0:
		with np.load(pol_path_id_grid + "/grid_modname_{}_genfiles.npz".format(og.mod_name)) as data:
			#X_all_ind_W_vals = data['X_all_ind_W_vals'].astype(np.float32)
			X_prime_vals = data['X_prime_vals']

	world_comm.barrier()
	if world_comm.rank == 0:
		print("Loaded interpolation points on all layer 2 masters.")

	# generate all the re-shaping functions
	reshape_out_pi, reshape_out_V, reshape_X_bar,\
		reshape_make_Apfunc_last, reshape_make_h_last,\
		reshape_vfunc_points, reshape_nadj_RHS, reshape_rent_RHS,\
		reshape_adj_RHS, reshape_RHS_Vfunc_rev, reshape_RHS_UFB,\
		reshape_vfunc_points = gen_reshape_funcs(og)

	del og
	gc.collect()

	A_min = 1E-10


	
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
	def mort_FOC(m, UC_prime_func1D, UC_prime_M_func1D):
		"""Error of first order cond. of mortgage interior solution (equation x
		in paper)

		Parameters
		----------
		m : float64
				Mortgage liability taken into t+1, m_t+1

		UC_prime_func1D : 1D array
				Interpolant of cond. expec. of marginal u of wrt to a_t+1

		UC_prime_M_func1D : 1D array
				Interpolant of cond. expec. of marginal u wrt to m_t+1

		Returns
		-------
		error: float64
				RHS marginal u wrt to m_t+1 - RHS marginal u wrt to a_t+1

		Notes
		-----
		The marginal utilities are interpolants conditoned on end of
		time t information as "functions" of m_t+1. See notes for
		`UC_cond_all' for states the marginal utilities are conditioned
		on

		A solution to the FOCs gives the optimal unconstrained mortgage
		choice m_t+1
		"""

		error = interp_as(M, UC_prime_func1D,np.array([m]))[0]\
					- interp_as(M, UC_prime_M_func1D,np.array([m]))[0]
		return error

	@njit
	def liq_rent_FOC_W(cons, h, q, alpha_housing, UC_prime_func1D):
		"""Solves frst order condition of liq. asset choice of renters
			(equation x in paper)

		Parameters
		----------
		cons : float64
				consumption at period t
		h : float64
				housing services consumed at t
		alpha_housing : float64
				housing share parameter at t 
		UC_prime_func1D : 1D array
				Interpolant of discounted cond. expec. of marginal u of wrt to a_t+1

		Returns
		-------
		a_prime : float64
				  Optimal interior solution a_t+1 implied by FOC

		Notes
		-----

		The marginal utility is an interpolant conditoned on end of
		time t information as "function" of a_t+1. See notes for
		`UC_cond_all' for the states the marginal utility is conditioned
		on

		The RHS marginal utility assumes h_t taken into
		period t+1 is H_min 
		"""

		LHS_val = uc(cons, h, alpha_housing)
		a_prime = interp_as(UC_prime_func1D, A, np.array([LHS_val]))[0]

		return a_prime

	@njit
	def HA_FOC(x_prime,
			   a_dc,
			   h,
			   q,
			   m,
			   alpha_housing,
			   beta,
			   wage,
			   mort_func1D,
			   UC_prime_func2D,
			   UC_prime_H_func2D,
			   UC_prime_M_func2D,
			   t,
			   ret_cons = False):
		"""Returns Euler error of housing first order condition
		(Equation x in paper) where liq asset FOC does not bind

		Parameters
		----------
		x_prime : float64
				a_t+1, t+1 liqud assett 
		a_dc : float64
				a_DC_t+1, DC assets (before t+1 returns)
		h : float64
				h_t, housing stock at t (recall h_t is consumed at t)
		q : float64
				P_t, house price at t
		m : float64
				M_t(1 + r_m), mort liability at t after time t interest
		beta : float64
				time t discount rate
		alpha_housing : float64
				time t housing share
		wage : float64
				wage at time t 
		mort_func1D : 1D array
				Optimal unconstrained M_t+1 as function of a_t+1
		UC_prime_func2D : 2D array
				t+1 RHS of Euler equation wrt a_t+1
		UC_prime_H_func2D : 2D array
				t+1 RHS of Euler equation wrt h_t
		UC_prime_M_func2D : 2D array
				t+1 RHS of Euler equation wrt to mortages
		t : int
				Age
		ret_cons :Bool
				if True, returns c_t for with  x_prime and implicit
				m_prime

		Returns
		-------
		Euler error : float64

		Notes
		-----

		The marginal utilities are interpolants conditoned on end of
		time t information as "function" of a_t+1 and m_t+1

		See notes for  `UC_cond_all' for states the marginal utility
		is be conditioned on

		Todo
		----

		Verify if `ref_const_FOC' should be inside or outside the
		absolute value sign

		"""

		max_loan = min(
			q * h * (1 - phi_c), phi_d * wage / amort_rate(t - 1))

		uc_prime_maxloan = max(1e-250, beta * eval_linear(X_cont_WAM2,
											  UC_prime_func2D,
											  np.array([x_prime,h, max_loan]), xto.LINEAR))

		m_prime_adj = eval_linear(X_cont_AH,
									mort_func1D,
									np.array([x_prime,h]),xto.LINEAR)
		m_prime = max(1e-250, min(m_prime_adj, max_loan))
		uc_prime_adj = max(1e-250,beta * eval_linear(X_cont_WAM2,
										  UC_prime_func2D,
										  np.array([x_prime,h, m_prime]),xto.LINEAR))

		c_t = max(C_min, uc_inv(uc_prime_adj, h, alpha_housing))

		# Indictor for whether actual constrained mortage is
		# constrained by collat. constraint
		ref_const_H = 0

		if m_prime == q * h * (1 - phi_c):
			ref_const_H = 1

		# interp values of RHS of housing Euler
		UC_prime_RHS = beta * eval_linear(X_cont_WAM2, UC_prime_func2D,
										  np.array([x_prime,h, m_prime]),xto.LINEAR)
		UC_prime_H_RHS = beta * eval_linear(X_cont_WAM2, UC_prime_H_func2D,
											np.array([x_prime,h, m_prime]),xto.LINEAR)


		# Calculate the marginal value of relaxising the collateral
		# constraint (ref_const_FOC)
		# Recall this is marginal utility of consumption today
		# - marginal cost of borrowing today (RHS of Euler wrt to m_t+1)
		# We only care about this value if it is positive i.e.
		# for negative values, the collat constraint will never binds
		ref_const_FOC = ref_const_H\
			* q * (1 - phi_c)\
			* (max(uc(c_t, h, alpha_housing)
				   - uc_prime_maxloan, 0))

		RHS = uc(c_t, h, alpha_housing) * q * (1 + tau_housing)\
			- UC_prime_H_RHS

		if ret_cons == True:
			return c_t
		else:

			return uh(c_t, h, alpha_housing) - RHS

	@njit
	def H_FOC(c, x_prime,
			  a_dc,
			  h,
			  q,
			  m,
			  alpha_housing,
			  beta,
			  wage,
			  mort_func1D,
			  UC_prime_func2D,
			  UC_prime_H_func2D,
			  UC_prime_M_func2D,
			  t):
		"""Returns Euler error of housing FOC with liquid assett FOC binding

			(Equation x in paper)

		Parameters
		----------
		c : float64
		 consumption at time t
		x_prime : float64
		  a_t+1 next period liquid
		a_dc : float64
		  before returns t+1 DC assets
		h : float64
		  H_t
		q : float64
				P_t house price
		m : float64
				M_t+1 mortgage liability
		mort_func1D : 1D array
				Optimal unconstrained M_t+1 as function of x_prime
		UC_prime_func2D : 2D array
				t+1 RHS of Euler equation wrt a_t+1
		UC_prime_H_func2D : 2D array
				t+1 RHS of Euler equation wrt H_t
		UC_prime_HFC_func2D : 2D array
				fixed cost marginal of t+1 housing adjustment
		UC_prime_M_func2D : 2D array
				t+1 RHS of Euler equation wrt to mortages
		t : int
				Age

		Returns
		-------
		Euler error : float64

		Notes
		-------

		See note for `HA_FOC' regarding
		"""

		# Binding liquid asset FOC binding implies
		# binding mortgage FOC, first calculate binding point

		max_loan = min(q * h * (1 - phi_c),
					   phi_d * wage / amort_rate(t - 1))
		uc_prime_fullpay = beta * eval_linear(X_cont_WAM2,
											  UC_prime_func2D,
											  np.array([x_prime,h, 0]),xto.LINEAR)
		uc_prime_maxloan = beta * eval_linear(X_cont_WAM2,
											  UC_prime_func2D,
											  np.array([x_prime,h, max_loan]),xto.LINEAR)
		ucmprime_m_fullpay = beta * eval_linear(X_cont_WAM2,
												UC_prime_M_func2D,
												np.array([x_prime,h, 0]),xto.LINEAR)
		ucmprime_m_maxloan = beta * eval_linear(X_cont_WAM2,
												UC_prime_M_func2D,
												np.array([x_prime,h,
														  max_loan]),xto.LINEAR)

		if uc_prime_maxloan > ucmprime_m_maxloan:
			m_prime = max_loan
		else:
			m_prime = 0

		# Calcuate RHS of Euler at m_prime

		UC_prime_M_RHS = beta * eval_linear(X_cont_WAM2, UC_prime_M_func2D,
											np.array([x_prime,h, m_prime]),xto.LINEAR)
		UC_prime_RHS = beta * eval_linear(X_cont_WAM2, UC_prime_func2D,
										  np.array([x_prime,h, m_prime]),xto.LINEAR)
		UC_prime_H_RHS = beta * eval_linear(X_cont_WAM2, UC_prime_H_func2D,
											np.array([x_prime,h, m_prime]),xto.LINEAR)

		# Calcuate indictor for collat constraint holding
		ref_const_H = 0
		
		if m_prime == q * h * (1 - phi_c):
			ref_const_H = 1

		# Marginal value of relaxing collat constraint
		ref_const_FOC = ref_const_H\
			* q * (1 - phi_c)\
			* (max(0,
				   uc(c, h, alpha_housing) - UC_prime_M_RHS))

		RHS = uc(c, h, alpha_housing) * q * (1 + tau_housing)\
			- UC_prime_H_RHS


		return uh(c, h, alpha_housing) - RHS

	# Define functions that solve the policy functions

	@njit
	def eval_M_prime(UC_prime_func1D, UC_prime_M_func1D):
		"""Solve mortgage FOC for unconstrained mortgage solution."""

		UC_prime_m_max = UC_prime_func1D[-1]
		UC_prime_M_m_max = UC_prime_M_func1D[-1]
		UC_prime_m_min = UC_prime_func1D[0]
		UC_prime_M_m_min = UC_prime_M_func1D[0]

		if mort_FOC(0, UC_prime_func1D, UC_prime_M_func1D)\
				* mort_FOC(M[-1], UC_prime_func1D, UC_prime_M_func1D) < 0:
			return brentq(mort_FOC, 0, M[-1],
						  args=(UC_prime_func1D,
								UC_prime_M_func1D), disp=False)[0]
		elif UC_prime_m_max >= UC_prime_M_m_max:
			return M[-1]
		else:
			return 0

	@njit
	def eval_pol_mort_W(t, UC_prime, UC_prime_M):
		"""Evaluates unconstrained  mortgage policy function as function of
		H_t,DC_t+1, A_t+1 and exogenous states at t."""

		mort_prime = np.zeros(len(X_all_C_ind))
		UC_prime_func_ex = UC_prime.reshape(all_state_shape)
		UC_prime_M_func_ex = UC_prime_M.reshape(all_state_shape)

		for i in range(len(X_all_C_ind)):
			h, h_ind = H[int(X_all_C_ind[i][7])], int(X_all_C_ind[i][7])
			a_prime, ap_ind = A[int(X_all_C_ind[i][5])], int(X_all_C_ind[i][5])
			q, q_ind = Q[int(X_all_C_ind[i][8])], int(X_all_C_ind[i][8])

			E_ind = int(X_all_C_ind[i][1])
			alpha_ind = int(X_all_C_ind[i][2])
			beta_ind = int(X_all_C_ind[i][3])
			Pi_ind = int(X_all_C_ind[i][4])
			DB_ind = 0
			adc_ind = int(X_all_C_ind[i][6])
			h_ind = int(X_all_C_ind[i][7])
			q_ind = int(X_all_C_ind[i][8])

			UC_prime_func1D = UC_prime_func_ex[DB_ind,
											   E_ind, alpha_ind,
											   beta_ind, Pi_ind, ap_ind,
											   adc_ind, h_ind, q_ind, :]
			UC_prime_M_func1D = UC_prime_M_func_ex[DB_ind,
												   E_ind, alpha_ind, beta_ind,
												   Pi_ind, ap_ind, adc_ind,
												   h_ind, q_ind, :]
			mort_prime[i] = max(min(0, eval_M_prime(UC_prime_func1D,
										 UC_prime_M_func1D)), M[-1])

			if mort_prime[i] == np.nan:
				mort_prime[i] = 0


		mort_func = mort_prime.reshape((len(DB), grid_size_W,
										grid_size_alpha,
										grid_size_beta,
										len(Pi),
										grid_size_A,
										grid_size_DC,
										grid_size_H,
										grid_size_Q))

		return mort_func

	@njit
	def eval_rent_pol_W(UC_prime):
		"""Creates renter policy function as function of wealth_t, q_t, dc_t+1
		and exogenous states at t using endog grid method."""

		a_end_1 = np.zeros(len(X_all_B_ind))  # Endog grid for a_t
		UC_prime_func_ex = UC_prime.reshape(all_state_shape)

		for i in range(len(X_all_B_ind)):

			h, h_ind = H[int(X_all_B_ind[i][6])], int(X_all_B_ind[i][6])
			q, q_ind = Q[int(X_all_B_ind[i][7])], int(X_all_B_ind[i][7])

			E_ind = int(X_all_B_ind[i][1])
			alpha_ind = int(X_all_B_ind[i][2])
			beta_ind = int(X_all_B_ind[i][3])
			Pi_ind = int(X_all_B_ind[i][4])
			DB_ind = 0
			adc_ind = int(X_all_B_ind[i][5])

			beta = beta_hat[int(X_all_B_ind[i][3])]
			alpha_housing = alpha_hat[int(X_all_B_ind[i][2])]

			# get UC_prime as interpolant on a_t+1
			# next period housing and mortgage is zero
			UC_prime_func1D = beta * UC_prime_func_ex[DB_ind,
													  E_ind, alpha_ind,
													  beta_ind, Pi_ind, :,
													  adc_ind, 0, q_ind, 0]

			# calculate optimal consumption for h and q value
			c_t = max(C_min, ch_ser(h, alpha_housing, phi_r*q))

			# calculate optimal a_t+1
			a_prime_1 = liq_rent_FOC_W(c_t, h, q, alpha_housing,
									   UC_prime_func1D)
			a_prime = max(min(a_prime_1, A_max_W), A_min)

			# generate endogenous grid
			a_end_1[i] = c_t + a_prime + q * phi_r * h

		a_end_2 = np.copy(a_end_1).reshape((len(DB), grid_size_W,
											grid_size_alpha,
											grid_size_beta,
											len(Pi),
											grid_size_DC,
											int(grid_size_HS),
											grid_size_Q))

		# make H (housing services) the last ind of the endog asst grid
		a_end_3 = a_end_2.transpose((0, 1, 2, 3, 4, 5, 7, 6))
		# reshape so housing service vals are rows
		a_end_4 = np.copy(a_end_3).reshape((int(1 *
												grid_size_W *
												grid_size_alpha *
												grid_size_beta *
												len(Pi) *
												grid_size_DC *
												grid_size_Q),
											int(grid_size_HS),
											))
		# Empty grid for housing services policy function with
		# a_t indices as rows
		h_prime_func_1 = np.zeros((int(len(DB) * grid_size_W *
									   grid_size_alpha *
									   grid_size_beta *
									   len(Pi) *
									   grid_size_DC *
									   grid_size_Q), len(A)))

		# Interpolate optimal housing services on endogenous asset grid

		for i in range(len(h_prime_func_1)):

			a_end_clean = np.sort(a_end_4[i][a_end_4[i] > 0])
			hs_sorted = np.take(H_R[a_end_4[i] > 0],
								np.argsort(a_end_4[i][a_end_4[i] > 0]))

			if len(a_end_clean) == 0:
				a_end_clean = np.zeros(1)
				a_end_clean[0] = A_min
				hs_sorted = np.zeros(1)
				hs_sorted[0] = H_min

			h_prime_func_1[i, :] = interp_as(a_end_clean, hs_sorted, W_W)
			h_prime_func_1[i, :][h_prime_func_1[i, :] <= 0] = H_min

		h_prime_func_2 = h_prime_func_1.reshape((len(DB),
												 grid_size_W,
												 grid_size_alpha,
												 grid_size_beta,
												 len(Pi),
												 grid_size_DC,
												 grid_size_Q,
												 grid_size_A))

		h_prime_func_3 = h_prime_func_2.transpose((0, 1, 2, 3, 4, 7, 5, 6))

		return h_prime_func_3

	@njit
	def interp_pol_noadj(assets_l, etas, cons_l):
		"""Interpolates no adjust policy functions using from endogenous
		grids."""

		assets_reshaped = assets_l.reshape(all_state_shape)
		assets_reshaped = np.transpose(assets_reshaped,
									   (0, 1, 2, 3, 4, 6, 7, 8, 9, 5))
		assets_reshaped_1 = np.copy(assets_reshaped)\
			.reshape((int(len(X_all_hat_ind)
						  / grid_size_A), int(grid_size_A)))

		etas_reshaped = etas.reshape(all_state_shape)
		etas_reshaped = np.transpose(np.copy(etas_reshaped),
									 (0, 1, 2, 3, 4, 6, 7, 8, 9, 5))
		etas_reshaped_1 = np.copy(etas_reshaped)\
			.reshape((int(len(X_all_hat_ind)
						  / grid_size_A),
					  int(grid_size_A)))

		cons_reshaped = cons_l.reshape(all_state_shape)
		cons_reshaped = np.transpose(cons_reshaped,
									 (0, 1, 2, 3, 4, 6, 7, 8, 9, 5))
		cons_reshaped_1 = np.copy(cons_reshaped)\
			.reshape((int(len(X_all_hat_ind)
						  / grid_size_A), int(grid_size_A)))

		# generate empty grids to fill with interpolated functions
		Aprime_noadjust_1 = np.zeros((int(len(X_all_hat_ind)
										  / grid_size_A), int(grid_size_A)))
		etas_primes_1 = np.zeros((int(len(X_all_hat_ind)
									  / grid_size_A),
								  int(grid_size_A)))
		C_noadj_1 = np.zeros((int(len(X_all_hat_ind)
								  / grid_size_A), int(grid_size_A)))

		# Interpolate
		for i in range(len(assets_reshaped_1)):

			a_prime_points = np.take(
				A[~np.isnan(assets_reshaped_1[i])], np.argsort(assets_reshaped_1[i]))

			Aprime_noadjust_1[i] = interp_as(assets_reshaped_1[i],
											 a_prime_points,
											 A)
			Aprime_noadjust_1[i][Aprime_noadjust_1[i]<=0] = A_min
			Aprime_noadjust_1[i][Aprime_noadjust_1[i]==np.nan] = A_min

			c_prime_points = np.take(cons_reshaped_1[i][~np.isnan(assets_reshaped_1[i])],
									 np.argsort(assets_reshaped_1[i]))

			C_noadj_1[i] = interp_as(assets_reshaped_1[i],
									 c_prime_points,
									 A)
			C_noadj_1[i][C_noadj_1[i]==np.nan] = C_min
			C_noadj_1[i][C_noadj_1[i]<=0] = C_min

			eta_prime_points = np.take(etas_reshaped_1[i][~np.isnan(assets_reshaped_1[i])],
									   np.argsort(assets_reshaped_1[i]))

			etas_primes_1[i] = interp_as(assets_reshaped_1[i],
										 eta_prime_points,
										 A)
			etas_primes_1[i][etas_primes_1[i] == np.nan] = 0

		Aprime_noadjust_2 = Aprime_noadjust_1\
			.reshape(all_state_A_last_shape)
		etas_primes_2 = etas_primes_1\
			.reshape(all_state_A_last_shape)
		C_noadj_2 = C_noadj_1\
			.reshape(all_state_A_last_shape)

		Aprime_noadj = np.transpose(Aprime_noadjust_2,
									(0, 1, 2, 3, 4, 9, 5, 6, 7, 8))
		etas_noadj = np.transpose(etas_primes_2,
								  (0, 1, 2, 3, 4, 9, 5, 6, 7, 8))
		C_noadj = np.transpose(C_noadj_2,
							   (0, 1, 2, 3, 4, 9, 5, 6, 7, 8))

		return C_noadj, etas_noadj, Aprime_noadj

	@njit
	def interp_pol_adj(A_prime,
					   C,
					   eta_adj, 
					   wealth_bar):
		"""Interpolates no adjust policy functions using from endogenous
		grids."""

		A_prime_adj_reshape = reshape_make_h_last(A_prime)
		C_adj = reshape_make_h_last(C)
		eta_adj_reshape = reshape_make_h_last(eta_adj)
		wealth_bar_reshape = reshape_make_h_last(wealth_bar)

		# New pol funcs which will be interpolated over uniform
		# wealth grid
		assets_prime_adj_1 \
			= np.empty((int(len(X_W_bar_hdjex_ind) / grid_size_HS),
						int(grid_size_A)))
		H_prime_adj_1\
			= np.empty((int(len(X_W_bar_hdjex_ind) / grid_size_HS),
						int(grid_size_A)))
		c_prime_adj_1\
			= np.empty((int(len(X_W_bar_hdjex_ind) / grid_size_HS),
						int(grid_size_A)))
		eta_adj_prime_1\
			= np.empty((int(len(X_W_bar_hdjex_ind) / grid_size_HS),
						int(grid_size_A)))

		# Interpolate over uniform wealth grid (1D)
		for i in range(len(wealth_bar_reshape)):

			wealth_x = wealth_bar_reshape[i][~np.isnan(wealth_bar_reshape[i])]
			assets_clean = A_prime_adj_reshape[i][~np.isnan(
				wealth_bar_reshape[i])]
			assts_x = np.take(assets_clean, np.argsort(wealth_x))
			#print(assts_x)
			h_clean = H_R[~np.isnan(wealth_bar_reshape[i])]
			h_x = np.take(h_clean, np.argsort(wealth_x))

			
			c_clean = C_adj[i][~np.isnan(wealth_bar_reshape[i])]
			c_x = np.take(c_clean, np.argsort(wealth_x))
			
			eta_clean = eta_adj_reshape[i][~np.isnan(wealth_bar_reshape[i])]
			eta_x = np.take(eta_clean, np.argsort(wealth_x))

			wealth_xs = np.sort(wealth_x)

			assets_prime_adj_1[i] = interp_as(wealth_xs, assts_x, W_W)
			assets_prime_adj_1[i][assets_prime_adj_1[i] <= 0] = A_min
			assets_prime_adj_1[i][assets_prime_adj_1[i]  == np.nan] = A_min

			c_prime_adj_1[i] = interp_as(wealth_xs, c_x, W_W)
			c_prime_adj_1[i][c_prime_adj_1[i] <= 0] = C_min
			c_prime_adj_1[i][c_prime_adj_1[i]  == np.nan] = C_min

			H_prime_adj_1[i] = interp_as(wealth_xs, h_x, W_W)
			H_prime_adj_1[i][H_prime_adj_1[i] <= 0] = H_min
			H_prime_adj_1[i][H_prime_adj_1[i] == np.nan] = H_min

			eta_adj_prime_1[i] = interp_as(wealth_xs, eta_x, W_W)


		H_adj = H_prime_adj_1.reshape((len(DB), grid_size_W,
									   grid_size_alpha,
									   grid_size_beta,
									   len(Pi),
									   grid_size_DC,
									   grid_size_Q,
									   grid_size_M,
									   grid_size_A))
		C_adj = c_prime_adj_1.reshape((len(DB), grid_size_W,
									   grid_size_alpha,
									   grid_size_beta,
									   len(Pi),
									   grid_size_DC,
									   grid_size_Q,
									   grid_size_M,
									   grid_size_A))
		Aprime_adj = assets_prime_adj_1.reshape((len(DB), grid_size_W,
												 grid_size_alpha,
												 grid_size_beta,
												 len(Pi),
												 grid_size_DC,
												 grid_size_Q,
												 grid_size_M,
												 grid_size_A))

		eta_adj_prime = eta_adj_prime_1.reshape((len(DB), grid_size_W,
												 grid_size_alpha,
												 grid_size_beta,
												 len(Pi),
												 grid_size_DC,
												 grid_size_Q,
												 grid_size_M,
												 grid_size_A))

		return C_adj, H_adj, Aprime_adj, eta_adj_prime

	@njit
	def eval_policy_W_noadj(t,
							mort_func,
							UC_prime_func,
							UC_prime_M_func,
							VF,
							a,b):
		
		"""Time t worker policy function evaluation using EGM 
			for non-adjusting home owners

			EGM evauates time t liquid assets via inverse Euler 

		Parameters
		----------
		t : int
			 age
		mort_func: 9D array
			M_t+1 function
		UC_prime_func : flat 9D array
				t+1 expected value of u_1(c,h)
		UC_prime_M_func: flat 9D array
				t+1 RHS of Euler equation wrt to M_t
		VF: flat 9D array
			 t+1 RHS Value function (undiscounted)
		a: int
		b: int

		Returns
		-------



		Notes
		-----
		Expectation of UC_prime_func and VF defined on:

				- time t E, alpha, beta, pi, house price
				- time t+1 liquid assets (before returns)
				- time t+1 DC assets (before returns)
				- time t housing stock (taken into time t+1) before
						t+1 depreciation
				- time t house price
				- M_t+1 mortgage liability (befre t+1 interest)

		M_t+1 function defined on:
				- time t E, alpha, beta, pi, house price
				- time t+1 liquid assets (before returns)
				- time t+1 DC assets (before returns)
				- time t housing stock (taken into time t+1) before
						t+1 depreciation
				- time t house price
				- time t consumption

		No adjust pol. functions are defined on:

				- time t DC/DB, E, alpha, beta, pi, house price
				- time t liquid wealth (after returns and net wages)
				- time t+1 DC asset (before returns)
				- time t housing (after t depreciation)
				- time t house price
				- time t mortgage liability (after interest at t)

		Adjust pol. functions are defined on:

				- time t DC/DB, E, alpha, beta, pi, house price
				- time t+1 DC asset (before returns)
				- time t house price
				- time t liquid wealth (after returns, net wages house sale)

		Todo
		----

		Double check the math for the min_const_FOC and ref_const_FOC
		especially the sign of the constraints at the re-finance

		How does min_const_FOC work when housing is adjusted down?
		"""

		# gen. empty grids to fill with endogenous values of:
		#  - assets_l (liq assets inc. wage)
		#  - eta (continuation value )
		#  - consumption

		assets_l = np.zeros(len(X_all_hat_ind[a:b]))
		etas = np.zeros(len(X_all_hat_ind[a:b]))
		cons_l = np.zeros(len(X_all_hat_ind[a:b]))

		# Reshape t+1 functions so they can be indexed by current
		# period states and be made into a function of M_t+1
		UC_prime_func_ex = UC_prime_func.reshape(all_state_shape)
		V_func_ex = VF.reshape(all_state_shape)
		UC_prime_M_func_ex = UC_prime_M_func.reshape(all_state_shape)

		for i in range(a,b):
			# Loop through each exogenous grid point
			# recall h is H_t i.e. housing *after* depreciation at t
			# i.e. housing that goes into utility at t
			# a_prime is liquid asset taken into next period
			# q is period t price of housing 

			h = H[X_all_hat_ind[i][7]] * (1 - delta_housing)
			a_prime = A[X_all_hat_ind[i][5]]
			q = Q[X_all_hat_ind[i][8]]
			m = M[X_all_hat_ind[i][9]]
			a_dc = A_DC[X_all_hat_ind[i][6]]

			E_ind = X_all_hat_ind[i][1]
			alpha_ind = X_all_hat_ind[i][2]
			beta_ind = X_all_hat_ind[i][3]
			Pi_ind = X_all_hat_ind[i][4]
			DB_ind = 0
			a_ind = int(X_all_hat_ind[i][5])
			adc_ind = int(X_all_hat_ind[i][6])
			h_ind = int(X_all_hat_ind[i][7])
			q_ind = int(X_all_hat_ind[i][8])

			beta = beta_hat[beta_ind]
			alpha_housing = alpha_hat[alpha_ind]
							   
			# get the next period U_prime values
			mfunc_ucprime = UC_prime_func_ex[DB_ind,
											 E_ind, alpha_ind,
											 beta_ind, Pi_ind,
											 :, adc_ind, h_ind,
											 q_ind, :]
			mfunc_uc_v_prime = V_func_ex[DB_ind,
												  E_ind, alpha_ind,
												  beta_ind, Pi_ind,
												  :, adc_ind, h_ind,
												  q_ind, :]

			mfunc_uc_m_prime = UC_prime_M_func_ex[DB_ind,
												  E_ind, alpha_ind,
												  beta_ind, Pi_ind,
												  :, adc_ind, h_ind,
												  q_ind, :]
			mort_func1D = mort_func[DB_ind, E_ind,
									alpha_ind,
									beta_ind, Pi_ind,
									:, adc_ind, h_ind,
									q_ind]

			# calculate consumption by checking whether non-negative 
			# mortgage consraint binds 

			# RHS with full payment
			max_pay_points = np.array([a_prime, 0])
			ucmprime_m_fp = beta * eval_linear(X_cont_WAM,
											   mfunc_uc_m_prime,
											   max_pay_points, xto.LINEAR)
			uc_prime_fp = max(1e-250,beta * eval_linear(X_cont_WAM,
											 mfunc_ucprime,
											 max_pay_points, xto.LINEAR))


			# Step 1: check if constrainted by full re-payment of liability
			if uc_prime_fp <= ucmprime_m_fp:
				c_t = min(
					max(C_min, uc_inv(uc_prime_fp, h, alpha_housing)), C_max)
				m_prime = 0
				UC_prime_RHS = uc_prime_fp

			# Step 2: otherwise, eval unconstrained 
			# Note we are still constrained by collateral 
			else:
				m_prime_adj = interp_as(A, mort_func1D, np.array([a_prime]), extrap= False)[0]
				m_prime = max(0, min(m_prime_adj, h*(1-phi_c)))
				UC_prime_RHS = max(1e-250,beta * eval_linear(X_cont_WAM,
												  mfunc_ucprime,
												  np.array([a_prime, m_prime]), xto.LINEAR))

				c_t = min(C_max, max(C_min, uc_inv(
					UC_prime_RHS, h, alpha_housing)))

			mort_payment = max(0, m - m_prime)
			pts_noadj = np.array([a_prime, m_prime])
			
			# calculate t+ value function 
			V_RHS = beta * eval_linear(X_cont_WAM,
												mfunc_uc_v_prime,
												pts_noadj, xto.LINEAR)

			# Define the endogenous liq assets after
			# wage and after vol. cont.
			# note we are NOT dividing through by (1+r)
			assets_l[int(i-a)] = c_t + a_prime + mort_payment
			cons_l[int(i-a)] = c_t
			etas[int(i-a)] = u(c_t, h,alpha_housing) + beta * V_RHS

		# Re-shape and interpolate the no adjustment endogenous grids
		# we want rows to  index a product of states
		# other than A_prime.
		# vals in the columns are time t liquid asssets for A_t+1
		return cons_l, etas, assets_l

	def gen_RHS_noadj(t,
					  comm,
					  mort_func,
					  UC_prime,
					  UC_prime_M,
					  VF,
					  points_noadj):
		"""Function evaluates no adjust polices  for time t
			on the policy evaluation grid 

			Then the function interpolates the policies on the
			full time t grid points. 

		Parameters
		----------
		t: int
		comm: communicator class
		mort_func:
		UC_prime:
		UC_prime_H:
		UC_prime_HFC:
		UC_prime_M:
		VF: 
		points_noadj

		Returns
		-------
		noadj_vals: 
		C_noadj: 
		etas_noadj: 
		Aprime_noadj: 
		
		"""

		# Split evaluation of no adjust policies across workers 
		if comm.rank == 2:
			a = 0
			b = int(len(X_all_hat_ind)/3)
			cons_l_1, etas_1, assets_l_1 = eval_policy_W_noadj(
				t, mort_func, UC_prime, UC_prime_M, VF, a,b)
			comm.Send(cons_l_1, dest = 0, tag = 1233)
			comm.Send(etas_1, dest = 0, tag = 1244)
			comm.Send(assets_l_1, dest = 0, tag = 1255)

		elif comm.rank == 3:
			a = int(len(X_all_hat_ind)*2/3)
			b = int(len(X_all_hat_ind))
			cons_l_3, etas_3, assets_l_3 = eval_policy_W_noadj(
				t, mort_func, UC_prime, UC_prime_M, VF,a,b )

			comm.Send(cons_l_3, dest = 0, tag = 1333)
			comm.Send(etas_3, dest = 0, tag = 1344)
			comm.Send(assets_l_3, dest = 0, tag = 1355)

		elif comm.rank == 0: 
			a = int(len(X_all_hat_ind)/3)
			b = int(len(X_all_hat_ind)*2/3)
			cons_l_2, etas_2, assets_l_2 = eval_policy_W_noadj(
				t, mort_func, UC_prime, UC_prime_M, VF,a,b)
		
		if 	comm.rank == 0:
			a = 0
			b = int(len(X_all_hat_ind)/3)
			cons_l_1, etas_1, assets_l_1  = np.zeros(len(X_all_hat_ind[a:b])),\
											np.zeros(len(X_all_hat_ind[a:b])),\
											np.zeros(len(X_all_hat_ind[a:b]))
			a = int(len(X_all_hat_ind)*2/3)
			b = int(len(X_all_hat_ind))
			cons_l_3, etas_3, assets_l_3  = np.zeros(len(X_all_hat_ind[a:b])),\
											np.zeros(len(X_all_hat_ind[a:b])),\
											np.zeros(len(X_all_hat_ind[a:b]))
			comm.Recv(cons_l_1, source = 2, tag = 1233)
			comm.Recv(etas_1, source = 2, tag = 1244)
			comm.Recv(assets_l_1, source = 2, tag = 1255)

			comm.Recv(cons_l_3, source = 3, tag = 1333)
			comm.Recv(etas_3, source = 3, tag = 1344)
			comm.Recv(assets_l_3, source = 3, tag = 1355)

			assets_l = np.concatenate((assets_l_1, assets_l_2,assets_l_3), axis = None)
			etas = np.concatenate((etas_1, etas_2,etas_3), axis = None)
			cons_l = np.concatenate((cons_l_1, cons_l_2,cons_l_3), axis = None)

			# Interpolate on endogenous grid
			C_noadj, etas_noadj, Aprime_noadj\
				= interp_pol_noadj(assets_l, etas, cons_l)

			# Re-shape policy functions
			noadj_vals = gen_RHSeval_pol_noadj(reshape_nadj_RHS(C_noadj),
											   reshape_nadj_RHS(etas_noadj),
											   reshape_nadj_RHS(Aprime_noadj),
											   points_noadj)

			noadj_vals = noadj_vals.reshape((1,
											 grid_size_W,
											 grid_size_alpha,
											 grid_size_beta,
											 len(Pi),
											 grid_size_H,
											 grid_size_Q,
											 grid_size_M,
											 len(V),
											 grid_size_A,
											 grid_size_DC,
											 4))\
				.transpose((0, 1, 2, 3, 8, 4, 9, 10, 5, 6, 7, 11))\
				.reshape((x_all_ind_len, 4))

			return noadj_vals, C_noadj, etas_noadj, Aprime_noadj
		else:
			return None

	def gen_RHS_adj(t,
					mort_func,
					UC_prime,
					UC_prime_H,
					UC_prime_M,
					VF,
					points_adj):
		"""Function evaluates polices, then calls on function to create RHS
		interpoled policies and returns both."""

		C_adj, H_adj, Aprime_adj,eta_adj = eval_policy_W_adj(t,
													 mort_func,
													 UC_prime,
													 UC_prime_H,
													 UC_prime_M, VF)
		adj_vals = gen_RHSeval_pol_adj(reshape_adj_RHS(C_adj),
									   reshape_adj_RHS(H_adj),
									   reshape_adj_RHS(Aprime_adj),
									   reshape_adj_RHS(eta_adj),
									   points_adj)
		adj_vals = adj_vals.reshape((1, grid_size_W,
									 grid_size_alpha,
									 grid_size_beta, len(Pi),
									 grid_size_Q, grid_size_M,
									 len(V),
									 grid_size_A, grid_size_DC,
									 grid_size_H, 6))\
			.transpose((0, 1, 2, 3, 7, 4, 8, 9, 10, 5, 6, 11))\
			.reshape((x_all_ind_len, 6))
		return adj_vals, C_adj, H_adj, Aprime_adj

	def gen_RHS_rentponts(UC_prime, points_rent):

		H_rent = eval_rent_pol_W(UC_prime)
		rent_pols = reshape_rent_RHS(H_rent)
		rent_vals = gen_RHSeval_pol_rent(rent_pols, points_rent)
		rent_vals = rent_vals.reshape((1,
									   grid_size_W,
									   grid_size_alpha,
									   grid_size_beta,
									   len(Pi),
									   grid_size_Q,
									   len(V),
									   grid_size_A,
									   grid_size_DC,
									   grid_size_H,
									   grid_size_M,
									   2)).\
			transpose((0, 1, 2, 3, 6, 4, 7, 8, 9, 5, 10, 11)).\
			reshape((x_all_ind_len, 2))
		return rent_vals, H_rent

	@njit
	def eval_policy_W_adj(t, mort_func,
						  UC_prime_func,
						  UC_prime_H_func,
						  UC_prime_M_func, VF):
		"""Time t worker policy function evaluation with adjustment

		Parameters
		----------
		t : int
			 age
		UC_prime_func : flat 10D array
						 t+1 expected value of u_1(c,h)
		UC_prime_H_func: flat 10D array
						  t+1 RHS of Euler equation wrt to H_t
		UC_prime_M_func: flat 10D array
						  t+1 mortgage choice
		VF_func: flat 10D array
				  t+1 value function

		Returns
		-------
		C_adj: 9D array
				adjuster consumption policy
		H_adj: 9D array
				adjust housing policy
		Aprime_adj: 9D array
					adjust liquid assett policy
		eta_adj: 9D array
					adjuster continuation value 

		Expectation of UC_prime_func and VF defined on:

				- time t E, alpha, beta, pi, house price
				- time t mortgage liability (after t interest)
				- time t+1 liquid assets (before returns)
				- time t+1 DC assets (before returns)
				- time t housing stock (taken into time t+1) before
						t+1 depreciation

		Adjust pol. functions are defined on:
				- time t DC/DB, E, alpha, beta, pi, house price
				- time t mortgage liability (after t interest)
				- time t+1 DC asset (before returns)
				- time t house price
				- time t liquid wealth (after returns,  net wages house sale)
		"""

		# A_prime is adjustment policy for next period liquid (endogenous)
		# wealth_bar is current period endogenous wealth grid (endogenous)
		# C is current period consumption (endogenous)

		A_prime = np.zeros(int(len(X_W_bar_hdjex_ind)))
		wealth_bar = np.zeros(int(len(X_W_bar_hdjex_ind)))
		C = np.copy(A_prime)
		eta_adj = np.copy(A_prime)

		UC_prime_func_ex = UC_prime_func.reshape(all_state_shape)
		UC_prime_H_func_ex = UC_prime_H_func.reshape(all_state_shape)
		V_func_ex = VF.reshape(all_state_shape)
		UC_prime_M_func_ex = UC_prime_M_func.reshape(all_state_shape)

		for i in range(len(X_W_bar_hdjex_ind)):

			"""For each i, the indices in X_W_bar_hdjex_ind[i] are:

			0 - DB/DC
			1 - E     (previous period)
			2 - alpha (previous period)
			3 - beta  (previous period)
			4 - Pi    (previous period)
			5 - A_DC (before returns) taken into T_R
			6 - H at T_R (coming into state, before depreciation)
			7 - Q at T_R (previous period)
			8 - M_t  at time t (after t interest)
			"""

			E_ind, alpha_ind, beta_ind = X_W_bar_hdjex_ind[i][1],\
				X_W_bar_hdjex_ind[i][2],\
				X_W_bar_hdjex_ind[i][3]
			Pi_ind, DB_ind, adc_ind, m_ind = X_W_bar_hdjex_ind[i][4],\
				0,\
				int(X_W_bar_hdjex_ind[i][5]), \
				int(X_W_bar_hdjex_ind[i][8])
			h_ind, q_ind = int(X_W_bar_hdjex_ind[i][6]),\
				int(X_W_bar_hdjex_ind[i][7])

			h, q, a_dc, m = H_R[h_ind], Q[q_ind], A_DC[adc_ind], M[m_ind]

			beta = beta_hat[beta_ind]
			alpha_housing = alpha_hat[alpha_ind]
							
			# get functions for next period U_prime and VF values
			# as functions of A_prime, H_prime and  M_prime
			UC_prime_func2D = UC_prime_func_ex[DB_ind,
											   E_ind, alpha_ind,
											   beta_ind, Pi_ind,
											   :, adc_ind, :,
											   q_ind, :]

			UC_prime_H_func2D = UC_prime_H_func_ex[DB_ind,
												   E_ind, alpha_ind,
												   beta_ind, Pi_ind,
												   :, adc_ind, :,
												   q_ind, :]

			VF_func2D = V_func_ex[DB_ind,
									E_ind, alpha_ind,
									beta_ind, Pi_ind,
									:, adc_ind, :,
									q_ind, :]

			UC_prime_M_func2D = UC_prime_M_func_ex[DB_ind,
												   E_ind, alpha_ind,
												   beta_ind, Pi_ind,
												   :, adc_ind, :,
												   q_ind, :]
			
			# get function of M_prime as function of A_prime and H_t
			mort_func1D = mort_func[DB_ind, E_ind,
									alpha_ind,
									beta_ind, Pi_ind,
									:, adc_ind, :,
									q_ind]

			# Calc wage (for max mortgage)
			wage = y(t, E[E_ind])

			# Make tuple of args for FOC Euler error functions
			args_HA_FOC = (a_dc, h, q, m, alpha_housing, beta,
						   wage, mort_func1D,
						   UC_prime_func2D, 
						   UC_prime_H_func2D,
						   UC_prime_M_func2D,
						   t)

			args_eval_c = (A_min, a_dc, h, q, m, alpha_housing, beta,
						   wage, mort_func1D,
						   UC_prime_func2D, UC_prime_H_func2D, UC_prime_M_func2D, t)

			max_loan = min(min(q * h * (1 - phi_c),
						   phi_d * wage / amort_rate(t - 1)), M[-1])

			# check if there is an interior solution to
			# housing, liq assett and constrained mortgage FOC

			A_prime[i] = np.nan
			C[i] = np.nan
			wealth_bar[i] = np.nan

			if h == H_min:
				A_prime[i] = 1E-10
				C[i] = C_min
				wealth_bar[i] = A_min

			if HA_FOC(1e-10, *args_HA_FOC)\
					* HA_FOC(100, *args_HA_FOC) < 0:
				# if interior solution to a_t+1, calculate it
				A_prime[i] = max(brentq(HA_FOC, 1e-10, 100,
										args=args_HA_FOC, disp=False)[0],
								 A_min)

				C[i] = min(C_max, max(
					C_min,
					HA_FOC(
						A_prime[i],
						a_dc,
						h,
						q,
						m,
						alpha_housing,
						beta,
						wage,
						mort_func1D,
						UC_prime_func2D,
						UC_prime_H_func2D,
						UC_prime_M_func2D,
						t,
						ret_cons=True)))
				m_prime_adj = eval_linear(X_cont_AH, mort_func1D,
										np.array([A_prime[i],h]), xto.LINEAR)
				m_prime = max(0, min(m_prime_adj, max_loan))
				mort_payment = m - m_prime

				# end. cash at hand wealth is defined as:
				# (1+r)a_t + net wage 
				# + P_t(1-delta)H_t-1

				wealth_bar[i] = C[i] + A_prime[i]\
					+ q * h * (1 + tau_housing)\
					+ mort_payment
				
				eta_adj[i] = u(C[i], h,alpha_housing) +  eval_linear(X_cont_WAM2,\
														 VF_func2D,\
														np.array([A_prime[i],h,m_prime]),\
														xto.LINEAR)

			# if no interior solution with liq asset unconstrainted,
			# check if interior solution to housing and constrained
			# mortage with binding liquid assett FOC at A_min
			elif H_FOC(C_min, *args_eval_c)\
					* H_FOC(C_max, *args_eval_c) < 0:
				C_at_amin = min(C_max, max(brentq(H_FOC, C_min, C_max,
												  args=args_eval_c, disp=False)[0], C_min))

				m_prime_adj_at_amin1 = eval_linear(X_cont_AH, mort_func1D,
												 np.array([A_min,h]),xto.LINEAR)
				m_prime_at_amin = max(0,min(m_prime_adj_at_amin1, max_loan))
				UC_prime_RHS_amin = beta\
					* eval_linear(X_cont_WAM2,
								  UC_prime_func2D,
								  np.array([A_min, h,m_prime_at_amin]),xto.LINEAR)

				# if liquid assett const. does not satisfy
				# FOC, throw point out
				#if uc(C_at_amin, h, alpha_housing) > UC_prime_RHS_amin:
				A_prime[i] = A_min
				C[i] = C_at_amin
				mort_payment =  m - m_prime_at_amin
				wealth_bar[i] = C[i] \
						+ A_prime[i]\
						+ q * h * (1 + tau_housing)\
						+ mort_payment
				eta_adj[i] = u(C[i], h, alpha_housing)
				#else:
				#	pass

			# force include point with zero housing
			# if solution adjusted to housing
			# less than H_min

			else:
				pass

		# interpolate C, H and A_prime on endogenous wealth grid

		C_adj, H_adj, Aprime_adj, eta_adj = interp_pol_adj(A_prime, C, eta_adj, wealth_bar)

		return C_adj, H_adj, Aprime_adj, eta_adj

	# Functions to generate and condition the RHS of Euler equation
	@njit
	def gen_RHSeval_pol_adj(C_adj, H_adj, Aprime_adj, Etaprime_adj, points_adj):
		"""Evalutes policies for housing adjusters on all points in the full
		state-space grid."""

		# Empty adjuster policy grid
		adj_vals_small1 = np.empty((len(C_adj),
									int(grid_size_H * len(V)
										* grid_size_A
										* grid_size_DC), 6))

		for i in range(len(C_adj)):
			# Loop through each i \in ADJX: = DBxExAlphaxBetaxPixQxM
			# for each i \in ADJX, the adjuster policy func
			# is a function mapping Wealth(t)xA_DC (t+1) to
			# c_t, H_t, a_t+1;
			# for each i \in ADJX, there are V_txA_txA_DC_txH_t-1 points
			# to be evaluated. Thus for each i, the function evaluation
			# returns |VxAxA_DCxHx3| points. The function evaluation
			# takes 2 points: Wealth, A_DC(t+1) for each point on
			# VxAxA_DC(t)xH

			# The first three columns of adj_vals_small1 are
			# c_t, h_t, a_t+1


			adj_vals_small1[i, :, 0] = eval_linear(X_QH_W, C_adj[i],
												   points_adj[i], xto.LINEAR)
			adj_vals_small1[i, :, 1] = eval_linear(X_QH_W, H_adj[i],
												   points_adj[i], xto.LINEAR)
			adj_vals_small1[i, :, 2] = eval_linear(X_QH_W, Aprime_adj[i],
												   points_adj[i], xto.LINEAR)
			adj_vals_small1[i, :, 5] = eval_linear(X_QH_W, Etaprime_adj[i],
												   points_adj[i], xto.LINEAR)

			# the fourth column of adj_vals_small1 is
			# extra mortgage payment
			# = wealth (after min pay) - C_t - A_t+1 - H_t*(1+tau)*P_t
			adj_vals_small1[i, :, 3] = points_adj[i, :, 0]\
				- adj_vals_small1[i, :, 0]\
				- adj_vals_small1[i, :, 2]\
				- adj_vals_small1[i, :, 1]\
				* Q[X_adj_func_ind[i, 5]]\
				* (1 + tau_housing)

			adj_vals_small1[i, :, 4] = points_adj[i, :, 1]

		return adj_vals_small1

	@njit
	def gen_RHSeval_pol_noadj(C_noadj, etas_noadj, Aprime_noadj, points_noadj):
		"""Evalutes pol for housing non-adjusters on all points in the full
		state-space grid."""

		# Empty no-adjuster policy array
		noadj_vals_small1 = np.empty((len(C_noadj),
									  int(len(V) * grid_size_A
										  * grid_size_DC),
									  4))

		for i in range(len(C_noadj)):


			noadj_vals_small1[i, :, 0] = eval_linear(X_cont_W_hat,
													 C_noadj[i],
													 points_noadj[i], xto.LINEAR)
			noadj_vals_small1[i, :, 1] = eval_linear(X_cont_W_hat,
													 etas_noadj[i],
													 points_noadj[i], xto.LINEAR)
			noadj_vals_small1[i, :, 2] = eval_linear(X_cont_W_hat,
													 Aprime_noadj[i],
													 points_noadj[i], xto.LINEAR)

			noadj_vals_small1[i, :, 0][noadj_vals_small1[i, :, 0] <= 0] = C_min
			#noadj_vals_small1[i, :, 1][noadj_vals_small1[i, :, 1]<=0] = A_min
			noadj_vals_small1[i, :, 2][noadj_vals_small1[i, :, 2] <= 0] = A_min

			noadj_vals_small1[i, :, 3] = points_noadj[i, :, 0]\
				- noadj_vals_small1[i, :, 0]\
				- noadj_vals_small1[i, :, 2]

			# Recall for non-adjusters, extra-pay cannot be less than 0
			#noadj_vals_small1[i, :, 3][noadj_vals_small1[i, :, 3] <= 0] = 0

		return noadj_vals_small1

	@njit
	def gen_RHSeval_pol_rent(H_rent, points_rent):
		"""Evalutes pol for housing renters on all points in the full state-
		space grid."""

		# Empty array for renter pols
		rent_vals_small1 = np.empty((len(H_rent),
									 int(grid_size_H
										 * grid_size_M
										 * len(V)
										 * grid_size_A
										 * grid_size_DC),
									 2))

		for i in range(len(H_rent)):
			rent_vals_small1[i, :, 0] = eval_linear(X_QH_W,
													H_rent[i],
													points_rent[i], xto.LINEAR)
			rent_vals_small1[i, :, 1] = points_rent[i, :, 0]

		return rent_vals_small1

	@njit
	def comb_RHS_pol_rent_norent(t, noadj_vals,
								 adj_vals,
								 rent_vals,my_X_all_ind):
		"""Evaluates policy and marginal utility
			functions for non-adjusters,
			adjusters and renters on X_all_ind grid on
			full grid at time t

		Parameters
		----------
		t: int
			age
		noadj_vals: 4D array
		adj_vals: 5D array
		rent_vals: 3D array

		Returns
		-------
		h_prime_norent_vals: 1D array
							  H_t+1 for adjusters and non-adjusters
							  before t+1 depreciation 
		ucfunc_norent: 1D array
						t marginal utility of cons
						adj and non-adjusters
		ufunc_norent: 1D array
						t utility value
						adj and non-adjusters
		extra_pay: 1D array
					bool value if extra mort payment made

		etas_ind: 1D array
					bool value if housing adjustment made

		cannot_rent: 1D array
						bool value if renter cash at hand< 0
						(cash at hand after liquidating)
		ucfunc_rent: 1D array
					  time t marginal utility for renters
		no_rent_points_p: 3D array
							policy function for non-renters
		rent_points_p: 3D array 
							policy function for renters

	

		Todo
		----
		- Check if extra_pay is alwasys 1 if people no longer constrianed 
			by min payment if non-adjusting 
		
		Notes
		-----

		- For indices description of parameters, see doc for gen_RHS_UC_func
		- First (0) index for all returns is x_all_ind
		- no_rent_points_p index as:
			|DB(0)xE(1)xA(2)xB(3)xPi(4)xQ(9)|x|V(5)xA(6)xDC(7)xH(8)xM(10)|x points(11)
		- the final index for no_rent_points_p and rent_points_p is:
		   a_prime_norent_vals(0), adcprime(1) h_prime_norent_vals (2) m_prime (3)

		"""

		# Generate indicator function for non-adjuster (|eta|<=1)
		noadj_vals[:, 1] = np.where(np.abs(noadj_vals[:, 1]) <= 1, 1, 0)
		#print(adj_vals.shape)
		etas_noadj = noadj_vals[:, 1]
		etas_adj = adj_vals[:, 5]

		etas_ind = etas_noadj>etas_adj
		etas_ind = etas_ind.astype(np.int32)
		# Generate policies for non-renters
		# eta*no-adjust + (1-eta)*adjust
		# recall final 5th col of adj_vals is
		# A_DC prime vals (common for renters, adjusters \
		# and non-adjusters since it only depends on V_t
		h_prime_norent_vals = adj_vals[:, 1]
		alpha_housing_list = alpha_hat[my_X_all_ind[:, 2]]
		h_prime_norent_vals[etas_ind == 1] = H[my_X_all_ind[:, 8]
											   ][etas_ind == 1] * (1 - delta_housing)
		h_prime_norent_vals[h_prime_norent_vals <= H_min] = H_min

		c_prime_norent_vals = adj_vals[:, 0]
		c_prime_norent_vals[etas_ind == 1] = noadj_vals[:, 0][etas_ind == 1]

		c_prime_norent_vals[c_prime_norent_vals < C_min] = C_min
		c_prime_norent_vals[c_prime_norent_vals > C_max] = C_max

		a_prime_norent_vals = adj_vals[:, 2]
		a_prime_norent_vals[etas_ind == 1] = noadj_vals[:, 2][etas_ind == 1]
		a_prime_norent_vals[a_prime_norent_vals <= A_min] = A_min

		mort_pay = adj_vals[:, 3]
		mort_pay[etas_ind == 1] = noadj_vals[:, 3][etas_ind == 1]
		m_prime = M[my_X_all_ind[:, 10]] - mort_pay

		m_prime[m_prime > M[-1]] = M[-1]
		m_prime[m_prime<0] = 0

		adcprime = adj_vals[:, 4]

		# Renter polices
		h_prime_rent = rent_vals[:, 0]
		h_prime_rent[h_prime_rent <= H_min] = H_min
		c_prime_rent = ch_ser(h_prime_rent, alpha_housing_list,phi_r * Q[my_X_all_ind[:, 9]])
		c_prime_rent[c_prime_rent <= C_min] = C_min

		# Recall 2nd col. of rent vals is the cash at hand for renters
		# at t, thus A_t+1 = cash at hand - rent payments - consumption
		a_prime_rent = rent_vals[:, 1] - c_prime_rent \
			- h_prime_rent * phi_r * Q[my_X_all_ind[:, 9]]
		a_prime_rent[a_prime_rent <= A_min] = A_min

		# Cannot Switch to renting if cash at hand is zero after making
		# full mortgage payment
		cannot_rent = np.zeros(np.shape(rent_vals[:, 1]))

		cannot_rent[np.where(rent_vals[:, 1] == A_min)] = 1

		# Marginal utilis for renters and non-renters

		ucfunc_norent = uc_vec(c_prime_norent_vals,
							   h_prime_norent_vals, alpha_housing_list)
		ufunc_norent = u_vec(c_prime_norent_vals,
							 h_prime_norent_vals, alpha_housing_list)
		ucfunc_rent = uc_vec(c_prime_rent,
							 h_prime_rent, alpha_housing_list)
		ufunc_rent = u_vec(c_prime_rent,
						   h_prime_rent, alpha_housing_list)


		return h_prime_norent_vals,\
			a_prime_norent_vals,\
			ucfunc_norent,\
			ufunc_norent,\
			etas_ind,\
			cannot_rent,\
			ucfunc_rent,\
			ufunc_rent,\
			adcprime,\
			m_prime, \
			a_prime_rent

	@njit
	def gen_RHS_lambdas(t, VF,
						UC_prime_H, 
						UF_B,
						points_p):
		"""Generates RHS values of VF_t+1, Lambda_t+1 and Xi_t for renters and
		non-renters conditioned on time t states in X_all_ind."""

		# Generate empty arrays to fill with evaluated values
		Xi = np.empty((len(points_p), len(X_V_func_DP_vals[:, 0])))
		Lambda_B = np.empty((len(points_p), len(X_V_func_DP_vals[:, 0])))
		VF_B = np.empty((len(points_p), len(X_V_func_DP_vals[:, 0])))

		# The list of vol contributions rates will be common across each
		# i in the loop below
		v = X_V_func_DP_vals[:, 0]

		for i in range(len(points_p)):

			"""Each i of no_rent_points_p is a list of t+1 policies, indexed by
			the cartesian product of (V_t, A_t, A_DC_t, H_t-1,M_t)

			The product of grids (V_t, A_t, A_DC_t, H_t-1,M_t) given
			by X_V_func_DP_vals

			For each j in the cart product of
			(V_t, A_t, A_DC_t, H_t,M_t), the
			list of t+1 states is a tuple (A_t+1, A_DC_t+1, H_t,M_t+1)

			Recall the value function interpolants
			for t+1 are in the shape:
			|DBx ExAlphax BetaxPixQ|xA(t+1)xA_DC(t+1)xH(t)xM(t+1)x
			V_t+1xLambda_t+1

			The i elements of no_rent_points_p are ordered by
			the product of the following time t states:

			0 - DB/DC
			1 - E
			2 - alpha
			3 - beta
			4 - Pi
			5 - Q

			Thus, we loop over each i, and for each i, we interpolate
			VxAxA_DCxHxM points of (A_t+1, A_DC_t+1, H_t+1,M_t+1) on the
			value function for i.
			"""
			acc_ind = int(0)

			# get Pi and beta_t at t for i
			pi = X_V_func_CR_vals[i, 4]
			beta_t = X_V_func_CR_vals[i, 3]

			# Interpolate no renter value function
			# returns list of |V_txA_txA_DC_txH_txM_t|x2 points
			# first column is V_t+1
			vf_func_vals = beta_t * eval_linear(X_cont_W_bar,
												VF[i],
												points_p[i], xto.LINEAR)

			lam_func_vals = beta_t * eval_linear(X_cont_W_bar,
												 UC_prime_H[i],
												 points_p[i], xto.LINEAR)

			# check: should bequest value go here?
			VF_B[i, :] = UF_B[i, :] + vf_func_vals

			v_ind = np.zeros(np.shape(v))
			v_ind[np.where(v>0)]= 1
			v_ind = v_ind.astype(np.float64)
			pi_ind = 0

			if pi!=def_pi:
				pi_ind =1

			vol_cost = adj_v(t, X_V_func_DP_vals[:, 1]) * v_ind
			Lambda_B[i, :] = beta_t * lam_func_vals
			Xi[i, :] = VF_B[i, :]\
				- vol_cost\
				- adj_pi(t, X_V_func_DP_vals[:, 2],
						 adj_p(17))*pi_ind

		return Xi, VF_B, Lambda_B

	def gen_RHS_UC_func(comm, t,
						VF,
						UC_prime_H, 
						A_prime,
						noadj_vals,
						adj_vals,
						rent_vals
						):
		"""Generate the unconditioned marginal utilities on a full grid
		 grid for time t.

		The marginal utilities hese will be *integrands*
		in the conditional expec. of the RHS of the
		Euler equation at t-1

		The marginal utilities are defined on
		the state conditioned on the discrete
		pension choices at time t

		Parameters
		----------
		comm: communicator class 
		t: int
			age
		VF: flat 10D array 
			 value function for time t+1
		Lambda: flat 10D array
				marginal value of DC t+1
		A_prime: 
				bequest values 
		noadj_vals: 4D array
					policy functions for non-adjusters
		adj_vals: 5D array
					policy functions for adjusters
		rent_vals: 3D array
					policy functions for renters

		Returns
		----------
		UC_prime_B:         11D array
		UC_prime_H_B:       11D array
		UC_prime_HFC_B:     11D array
		Lambda_B:           11D array
		Xi:                 11D array

		Notes
		-----
		
		- Lambda, VF are indexed by X_all_hat_ind
		i.e. are indexed by all state-points except
		V. The DC index is for end of t period
		DC assets, DC_t+1, before t+1 returns
		
		- noadj_vals, adj_vals and rent vals are indexed by 
		  all indices and a policy index.
		-  Index (policy index) for adjusters:
		   (0) Index of X_all_ind
		   (1) C
		   (2) H_t+1 (adjusters)
		   (3) A_t+1
		   (4) Extra mortgage payment 
		   (5) A_DC_t+1
		- Final index for non-adjusters:
			(0) Index of X_all_ind
			(1) C
			(2) eta_t
			(3) A_t+1
			(4) Extra mortgage payment 
		- Final index for renters:
			(0) Index of X_all_ind
			(1) HS_t
			(2) renter cash at hand after all assets
				and mortgage liquidated

		"""

		# Scatter adj, noadj and rent grid vals to all 4 cores

		my_N_shape = (x_all_ind_len_a[0]//comm.size, 6)

		if comm.rank == 0:
			noadj_vals_chunks = np.array_split(noadj_vals,comm.size,axis=0)
			adj_vals_chunks = np.array_split(noadj_vals,comm.size,axis=0)
			rent_vals_chunks = np.array_split(noadj_vals,comm.size,axis=0)
			X_all_ind_split = np.array_split(X_all_ind,comm.size,axis=0)
		else:
			noadj_vals_chunks = None
			adj_vals_chunks = None
			rent_vals_chunks = None
			X_all_ind_split = None

		
		if comm.rank == 0:
			h_prime_norent_vals = np.ascontiguousarray(np.empty(len(noadj_vals)))
			a_prime_norent_vals = np.ascontiguousarray(np.empty(len(noadj_vals)))
			uc_prime_norent = np.ascontiguousarray(np.empty(len(noadj_vals)))
			u_prime_norent = np.ascontiguousarray(np.empty(len(noadj_vals)))
			etas_ind = np.ascontiguousarray(np.empty(len(noadj_vals)))
			cannot_rent  = np.ascontiguousarray(np.empty(len(noadj_vals)))
			uc_prime_rent = np.ascontiguousarray(np.empty(len(noadj_vals)))
			u_prime_rent  = np.ascontiguousarray(np.empty(len(noadj_vals)))
			adcprime = np.ascontiguousarray(np.empty(len(noadj_vals)))
			m_prime  = np.ascontiguousarray(np.empty(len(noadj_vals)))
			a_prime_rent= np.ascontiguousarray(np.empty(len(noadj_vals)))
		else:
			h_prime_norent_vals = None
			a_prime_norent_vals = None
			uc_prime_norent = None
			etas_ind = None
			cannot_rent  = None
			uc_prime_rent = None
			u_prime_rent  = None
			adcprime = None
			m_prime  = None
			a_prime_rent= None 
			u_prime_norent =None

		my_noadj_vals = comm.scatter(noadj_vals_chunks, root =0)
		my_adj_vals = comm.scatter(noadj_vals_chunks, root =0)
		my_rent_vals = comm.scatter(noadj_vals_chunks, root =0)
		my_X_all_in = comm.scatter(X_all_ind_split, root =0)




		my_h_prime_norent_vals,\
		my_a_prime_norent_vals,\
		my_ucfunc_norent,\
		my_ufunc_norent,\
		my_etas_ind,\
		my_cannot_rent,\
		my_ucfunc_rent,\
		my_ufunc_rent,\
		my_adcprime,\
		my_m_prime,my_a_prime_rent  = comb_RHS_pol_rent_norent(t, my_noadj_vals,
													my_adj_vals,
													my_rent_vals, my_X_all_in)

		sendcounts = np.array(comm.gather(len(my_h_prime_norent_vals), 0))


		
		comm.Gatherv(np.ascontiguousarray(my_h_prime_norent_vals), recvbuf=(h_prime_norent_vals, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_a_prime_norent_vals), recvbuf=(a_prime_norent_vals, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_ucfunc_norent), recvbuf=(uc_prime_norent, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_ufunc_norent), recvbuf=(u_prime_norent, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_etas_ind), recvbuf=(etas_ind, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_cannot_rent), recvbuf=(cannot_rent, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_ucfunc_rent), recvbuf=(uc_prime_rent, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_ufunc_rent), recvbuf=(u_prime_rent, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_adcprime), recvbuf=(adcprime, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_m_prime), recvbuf=(m_prime, sendcounts), root =0)
		comm.Gatherv(np.ascontiguousarray(my_a_prime_rent), recvbuf=(a_prime_rent, sendcounts), root =0)

		
		if comm.rank == 0:
			no_rent_points_p = reshape_vfunc_points(
				a_prime_norent_vals, adcprime, h_prime_norent_vals, m_prime)

			rent_points_p = reshape_vfunc_points(a_prime_rent, adcprime, np.full(
				len(X_all_ind), H_min), np.full(len(X_all_ind), 0))

		if comm.rank == 0:
			comm.Send(no_rent_points_p, dest=1, tag=222)
			comm.Send(u_prime_norent, dest=1, tag=221)

		# Generate the time t policies interpolated
		# on X_all_ind grid

		# value funcs are indexed by
		# DB(0)xE(1)xA(2)xB(3)xPi(4)xA(5)xA_DC(6)xH(7)xQ(8)xM(9)
		# re-order to
		# DB(0)xE(1)xA(2)xB(3)xPi(4)xQ(8)xA(5)xA_DC(6)xH(7)xM(9)

		# Extra_pay is vector of mortgage payments in addition
		# the min. amort payment
		# Recall that all renters are not min payment
		# constrained as they have to pay off thier mortgage
		#extra_pay[extra_pay < 0] = 0
		#extra_pay = extra_pay > 1e-5

		# t+1 period mortgage balance for non-renters

		# time t utility vector for  all X_all_ind states

		# Generate t+1 continuous states given policies
		# these are used to evaluate tiem t conditional value function
		# and Lamba function

		VF = VF.reshape(all_state_shape)\
			.transpose((0, 1, 2, 3, 4, 8, 5, 6, 7, 9))\
			.reshape((v_func_shape))

		UC_prime_H = UC_prime_H.reshape(all_state_shape)\
			.transpose((0, 1, 2, 3, 4, 8, 5, 6, 7, 9))\
			.reshape((v_func_shape))

		# Loop over all points of the states in  X_all_ind
		# interpolate values of value function and
		# Lambda function back to period t
		if comm.rank == 0:
			
			Xi_rent, VF_B_rent, Lambda_B_rent = gen_RHS_lambdas(t, VF,
																	UC_prime_H,
																reshape_RHS_UFB(
																	u_prime_rent),
																rent_points_p)
			Xi_rent = reshape_RHS_Vfunc_rev(Xi_rent)
			Lambda_B_rent = reshape_RHS_Vfunc_rev(Lambda_B_rent)
			VF_B_rent = reshape_RHS_Vfunc_rev(VF_B_rent)
			

		if comm.rank == 1:

			# Empty arrays to be filled by arrays received from rank 0
			no_rent_points_p = np.empty((int(1      * grid_size_W\
													* grid_size_alpha\
													* grid_size_beta\
													* len(Pi)\
													* grid_size_Q),
												int(len(V)\
													* grid_size_A\
													* grid_size_DC\
													* grid_size_H\
													* grid_size_M), 4))

			u_prime_norent = np.empty((int(1 * grid_size_W * grid_size_alpha
										   * grid_size_beta
										   * len(Pi) * grid_size_Q * len(V) * grid_size_A * grid_size_DC * grid_size_H * grid_size_M), 1))

			comm.Recv(no_rent_points_p, source=0, tag=222)
			comm.Recv(u_prime_norent, source=0, tag=221)
			Xi_norent, VF_B_norent, Lambda_B_norent = gen_RHS_lambdas(t, VF,
																		UC_prime_H, 
																	  reshape_RHS_UFB(
																		  u_prime_norent),
																	  no_rent_points_p)
			Xi_norent = reshape_RHS_Vfunc_rev(Xi_norent)
			Lambda_B_norent = reshape_RHS_Vfunc_rev(Lambda_B_norent)
			VF_B_norent = reshape_RHS_Vfunc_rev(VF_B_norent)
			comm.Send(Xi_norent, dest=0, tag = 999)
			comm.Send(Lambda_B_norent, dest=0, tag = 997)
			comm.Send(VF_B_norent, dest=0, tag = 996)
		
		if comm.rank == 0:

			Xi_norent = np.empty(np.shape(Xi_rent))
			Lambda_B_norent = np.empty(np.shape(Lambda_B_rent))
			VF_B_norent = np.empty(np.shape(VF_B_rent))

			comm.Recv(Xi_norent, source=1, tag=999)
			comm.Recv(Lambda_B_norent, source=1, tag=997)
			comm.Recv(VF_B_norent, source=1, tag=996)

			UF_B_norent = u_prime_norent
			UF_B_rent = u_prime_rent

			zeta = VF_B_rent / VF_B_norent
			#zeta = reshape_RHS_Vfunc_rev(zeta)

			# zeta is multiplier that indicates renter if zeta>1
			# cannot rent is an indictator that someone is not able
			# to pay off thier mortgage liability after renting decision
			# hence will not rent
			# renter_ind = zeta>0 AND cannot_rent == 0
			zeta[np.isnan(zeta)] = 0
			renter_ind = np.zeros(np.shape(zeta))
			renter_ind[np.where(zeta>1)] = 1
			renter_ind = renter_ind * (1 - cannot_rent)
			renter_ind[np.isnan(renter_ind)] = 0


			uc_prime = uc_prime_norent
			uc_prime[renter_ind == 1] = uc_prime_rent[renter_ind == 1]

			# RHS values of Euler equation
			# First, with respect to A_t
			UC_prime_B = (1 + r) * (s[int(t - 1)] * uc_prime + (1 - s[int(t - 1)])
									* b_prime(A_prime))
			# wrt to H_t-1
			UC_prime_H_B_rent = renter_ind*Q[X_all_ind[:, 9]]\
				* (1 - delta_housing - tau_housing )\
				* (s[int(t - 1)] * uc_prime
				   + (1 - s[int(t - 1)]) * b_prime(A_prime))


			UC_prime_H_B = (renter_ind + (1-etas_ind)*UC_prime_H_B_rent) + etas_ind*(1-delta_housing)*Lambda_B_norent

			# the fixed cost paif if H_t-1 is adjusted

			# wrt to mortgage m_t
			UC_prime_M_B = s[int(t - 1)] * uc_prime \
				+ (1 - s[int(t - 1)]) * b_prime(A_prime)

			# Return RHS value function, period utility
			# function and Lambda function values state by state
			# Xi function is the period utility + Lambda - adjustment cost

			VF_B = renter_ind * VF_B_rent\
				+ (1 - renter_ind) * VF_B_norent
			Xi = renter_ind * Xi_rent\
				+ (1 - renter_ind) * Xi_norent
			UF_B = renter_ind * UF_B_rent\
				+ (1 - renter_ind) * UF_B_norent
			del VF_B_norent, VF_B_rent, Xi_norent, UF_B_norent
			gc.collect()
			return UC_prime_B, UC_prime_H_B, UC_prime_M_B,\
					VF_B, Xi, UF_B, zeta.astype(np.float32).reshape(all_state_shape_hat)
		else:
			return 0

	# @njit
	def UC_cond_DC(UC_prime_B,
				   UC_prime_H_B,
				   UC_prime_M_B, VF_B,
				   Xi):
		"""Indicies of inputs are ordered by:

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
		"""
		# Reshape to condition out pi

		UC_prime_copi, UC_prime_H_copi,\
			UC_prime_M_copi, VF_B_copi, Xi_copi,\
			state_index_copi = \
			reshape_out_pi(UC_prime_B),\
			reshape_out_pi(UC_prime_H_B),\
			reshape_out_pi(UC_prime_M_B), reshape_out_pi(VF_B),\
			reshape_out_pi(Xi),\
			np.float64(reshape_out_pi(X_all_ind[:, 0]))

		DB_index_new = np.zeros(len(UC_prime_copi))
		scaling_pi = np.add(sigma_DB_pi * (1 - state_index_copi[:, 0]),
							sigma_DC_pi * (state_index_copi[:, 0])).astype(np.float64)
		DB_index_new = state_index_copi[:, 0]
		
		Xi_copi_temp = np.add(Xi_copi / scaling_pi[:, np.newaxis],
						 - np.max(Xi_copi / scaling_pi[:, np.newaxis],
								  axis=1)[:, np.newaxis])

		prob_pi = np.exp(Xi_copi_temp)\
			/ np.sum(np.exp(Xi_copi_temp), axis=1)[:, np.newaxis]

		prob_pi[np.where(np.isnan(prob_pi))] = .001
		UC_prime_cpi = einsum_row(prob_pi, UC_prime_copi)
		UC_prime_H_cpi = einsum_row(prob_pi, UC_prime_H_copi)
		UC_prime_M_cpi = einsum_row(prob_pi, UC_prime_M_copi)
		VF_B_cpi = einsum_row(prob_pi, VF_B_copi)
		Xi_cpi = einsum_row(prob_pi, Xi_copi)
		# Reshape to condition out V
		UC_prime_cov, UC_prime_H_cov,\
			UC_prime_M_cov, VF_B_cov, Xi_cov,\
			DB_index_V = reshape_out_V(UC_prime_cpi),\
			reshape_out_V(UC_prime_H_cpi),\
			reshape_out_V(UC_prime_M_cpi),\
			reshape_out_V(VF_B_cpi),\
			reshape_out_V(Xi_cpi),\
			reshape_out_V(DB_index_new)

		scaling_v = np.add(sigma_DB_V *
						   np.array((1 -
									 DB_index_V[:, 0])), sigma_DC_V *
						   np.array((DB_index_V[:, 0]))).astype(np.float64) 
		Xi_cov_temp = np.add(Xi_cov / scaling_v[:, np.newaxis],
						- np.max(Xi_cov / scaling_v[:, np.newaxis],
								 axis=1)[:, np.newaxis])
		prob_v = np.exp(Xi_cov_temp)\
			/ np.sum(np.exp(Xi_cov_temp), axis=1)[:, np.newaxis]

		prob_v[np.where(np.isnan(prob_v))] = .001

		UC_prime_cv = einsum_row(prob_v, UC_prime_cov)
		UC_prime_H_cv = einsum_row(prob_v, UC_prime_H_cov)
		UC_prime_M_cv = einsum_row(prob_v, UC_prime_M_cov)
		VF_B_cv = einsum_row(prob_v, VF_B_cov)
		Xi_cv = einsum_row(prob_v, Xi_cov)

		UC_prime_DCC, UC_prime_H_DCC, UC_prime_M_DCC, VF_DCC\
			= reshape_X_bar(UC_prime_cv),\
			reshape_X_bar(UC_prime_H_cv),\
			reshape_X_bar(UC_prime_M_cv),\
			reshape_X_bar(VF_B_cv)

		Xi_cov_out = Xi_cov.reshape(int(len(DB)), grid_size_W,
									grid_size_alpha,
									grid_size_beta,
									grid_size_A,
									grid_size_DC,
									grid_size_H,
									grid_size_Q,
									grid_size_M,
									len(V))
		Xi_copi_out = Xi_copi.reshape(int(len(DB)), grid_size_W,
									  grid_size_alpha,
									  grid_size_beta,
									  len(V),
									  grid_size_A,
									  grid_size_DC,
									  grid_size_H,
									  grid_size_Q,
									  grid_size_M,
									  len(Pi))
		del Xi_copi_temp
		del Xi_cov_temp
		gc.collect()
		return UC_prime_DCC, UC_prime_H_DCC, UC_prime_M_DCC, VF_DCC, prob_v.reshape(prob_v_shape), prob_pi.reshape(prob_pi_shape), Xi

	@njit
	def UC_cond_all(t, UC_prime_DCC, UC_prime_H_DCC, UC_prime_M_DCC, VF_DCC):
		"""Generate RHS t+1 Euler equation conditioned on:

				- housing stock taken into time t+1 
				- DC assets (before returns) taken into time t+1
				- mortgage liability taken into time t+1 (before interest at t+1)
				- t period housing stock (housing stock at start of t)
				- liquid assets taken into time t+1 (before returns)
				- t wage, alpha and beta shock and risk choice Pi at t
				- t period house price
				- DB/DC

		Parameters
		----------
		UC_prime_DCC: 1D array (10D array)
		UC_prime_H_DCC: 1D array (10D array)
		UC_prime_HFC_DCC: 1D array (10D array)
		UC_prime_M_DCC: 1D array (10D array
		Lambda_DCC: 1D array (10D array)
		VF_DCC: 1D array (10D array)
		UF_DCC: 1D array (10D array)

		Returns
		-------
		UC_prime: flat 10D array
		UC_prime_H: flat 10D array
		UC_prime_HFC: flat 10D array
		UC_prime_M: flat 10D array
		Lambda: flat 10D array
		VF: flat 10D array
		UF: flat 10D array

		Todo
		----
		Check in the math that the use of AD_return_prime
		here to multiply the Lambda function is correct

		"""
		UC_prime = np.empty(len(X_all_hat_ind))
		UC_prime_H = np.empty(len(X_all_hat_ind))
		UC_prime_M = np.empty(len(X_all_hat_ind))
		VF = np.empty(len(X_all_hat_ind))

		for i in range(len(X_all_hat_ind)):

			"""
			Indicies along each row of X_all_hat_ind are:

			0 - DB/DC (this is always 0 since each solver solves only for DB/DC)
			1 - E     (time t)
			2 - alpha (time t)
			3 - beta  (time t)
			4 - Pi    (time t)
			5 - A (before returns) at t+1
			6 - A_DC (before returns) taken into t+1
			7 - H at t (coming into state, BEFORE depreciation)
			8 - Q at t 
			9 - M at t
			"""
			E_ind = X_all_hat_ind[i][1]
			alpha_ind = X_all_hat_ind[i][2]
			beta_ind = X_all_hat_ind[i][3]
			DB_ind = int(0)
			a_ind = int(X_all_hat_ind[i][5])
			adc_ind = int(X_all_hat_ind[i][6])
			m_ind = int(X_all_hat_ind[i][9])
			h_ind = int(X_all_hat_ind[i][7])

			# A_DC_{t+1}, Q_{t+1}, M_{t+1} will be subject to shocks in t+1
			# pull out values of A_DC_{t+1}(1 +R_DC), Q_{t+1} and M_{t+1}(1+r_m)
			# from X_prime_vals

			point = X_prime_vals[i][:, 0:3]

			R_m_prime = X_prime_vals[i][:, 2] / M[m_ind]
			R_m_prime[np.where(np.isnan(R_m_prime))] = 1.03
			ADC_return_prime = X_prime_vals[i][:, 0] / A_DC[adc_ind]

			E_ind_p, alpha_ind_p, beta_ind_p\
				= E_ind, alpha_ind, beta_ind

			#print(R_m_prime)

			# gen values of t+1 unconditioned on t expectation of the iid shocks
			# for DC, Mortgage and house price
			U1, U2, U3, V\
				= eval_linear(X_DCQ_W,
							  UC_prime_DCC[DB_ind,
										   E_ind_p,
										   alpha_ind_p,
										   beta_ind_p,
										   a_ind, :,
										   h_ind, :, :],
							  point, xto.LINEAR),\
				eval_linear(X_DCQ_W,
							UC_prime_H_DCC[DB_ind,
										   E_ind_p,
										   alpha_ind_p,
										   beta_ind_p,
										   a_ind, :,
										   h_ind, :, :],
							point, xto.LINEAR),\
				eval_linear(X_DCQ_W,
							UC_prime_M_DCC[DB_ind,
										   E_ind_p, alpha_ind_p,
										   beta_ind_p, a_ind, :,
										   h_ind, :, :],
							point, xto.LINEAR) * R_m_prime,\
				eval_linear(X_DCQ_W,
							VF_DCC[DB_ind, E_ind_p,
								   alpha_ind_p, beta_ind,
								   a_ind, :, h_ind, :, :],
							point, xto.LINEAR)

			UC_prime[i], UC_prime_H[i],\
				UC_prime_M[i], VF[i]\
				= d0(U1, Q_DC_P), d0(U2, Q_DC_P),\
				d0(U3, Q_DC_P), d0(V, Q_DC_P)

		UC_prime[np.where(UC_prime<= 0)] = 1E-250
		UC_prime_H[np.where(UC_prime_H<= 0)] = 1E-250

		UC_prime_M[np.where(UC_prime_M<=0)] = 1E-250

		UC_prime[np.where(np.isnan(UC_prime))]= 1E-250
		UC_prime_H[np.where(np.isnan(UC_prime_H))]= 1E-250
		UC_prime_M[np.where(np.isnan(UC_prime_M))]= 1E-250

		return UC_prime, UC_prime_H, UC_prime_M, VF

	
	def solve_LC_model(comm,\
						load_ret):
		"""  Function to solve worker policies via backward iteration
		"""

		# Step 1:
		# Set path for loading retiree pols
		# If no retiree policies are calculated, 
		# they need to be loaded from the parent folder 
		# with mod_name. 
		#
		# Currently, policies need to be
		# manually pasted in the mod_name folder
		# If retiree policiese are solved, they are 
		# saved in the pol_path_id folder.

		if load_ret == 1:
			ret_path_load = scr_path
		else: 
			ret_path_load = pol_path_id

		if load_ret == 0 and comm.rank == 0:
			# Generate retiree policies 
			UC_prime, UC_prime_H,UC_prime_HFC, UC_prime_M, Lambda, VF\
														 = gen_R_pol()
			ret_pols\
			 = (UC_prime, UC_prime_H, UC_prime_HFC, UC_prime_M, Lambda, VF)
			pickle.dump(ret_pols, open("{}/ret_pols_{}.pol".format(pol_path_id, str(acc_ind[0])), "wb"))
			gc.collect()
		else:
			pass 

		comm.barrier()
		if load_ret == 1 or comm.rank == 1 or comm.rank == 2 or comm.rank == 3:
			# Load retiree policies on all ranks
			ret_pols = pickle.load(open("{}/ret_pols_{}.pol"\
										.format(ret_path_load, str(acc_ind[0])), "rb"))
			(UC_prime, UC_prime_H, UC_prime_HFC,UC_prime_M, Lambda, VF)\
										= ret_pols

		UF = VF
		start1 = time.time()
		np.seterr(divide='ignore') 
		for Age in np.arange(int(tzero), int(R))[::-1]:
			start2 = time.time() 
			# Load RHS interpolation  points for age from scratch drive 
			with np.load(pol_path_id_grid + "/grid_modname_{}_age_{}.npz"\
								.format(mod_name, Age)) as data:
				if comm.rank == 0 or comm.rank == 2 or comm.rank == 3:
					points_noadj_vec = data['points_noadj_vec']
					points_rent_vec = data['points_rent_vec']
					A_prime = data['A_prime']
				else:
					points_adj_vec = data['points_adj_vec']

			start = time.time()
			t = Age
			
			# Step 1: Evaluate optimal mortgage choice func
			mort_func = eval_pol_mort_W(t, UC_prime, UC_prime_M)
			print(x_all_ind_len_a[0])

			# Step 2: Evaluate renter and non- adj pol and points on rank 0,2,3
			if comm.rank == 0 or comm.rank == 2 or comm.rank == 3:
				if verbose == True:
					print("Solving for age_{}".format(Age))
				noadjpolsoutparr =\
				gen_RHS_noadj(t,\
								comm,\
								mort_func,\
								UC_prime,\
								UC_prime_M,\
								VF,\
								points_noadj_vec)
				if comm.rank == 0:
					noadj_vals,C_noadj, etas_noadj, Aprime_noadj = noadjpolsoutparr
					rent_vals, H_rent = gen_RHS_rentponts(UC_prime,\
										  points_rent_vec)
					adj_vals  = np.empty((len(X_all_ind), 6))

			# Step 3: Evaluate policy and points for housing adjusters on rank 1
			if comm.rank == 1:
				adj_vals,C_adj, H_adj, Aprime_adj = gen_RHS_adj(t,
											mort_func,
											UC_prime,
											UC_prime_H,
											UC_prime_M, 
											VF,
											points_adj_vec)

				# Send results to rank 0
				comm.Send(adj_vals, dest= 0, tag= 4)
				comm.Send(C_adj, dest= 0, tag= 5)
				comm.Send(H_adj,dest= 0, tag= 6)
				comm.Send(Aprime_adj, dest= 0, tag= 7)
				A_prime = 0
				noadj_vals = np.empty((x_all_ind_len_a[0], 6))
				rent_vals = np.empty((x_all_ind_len_a[0], 6))
			
			if comm.rank !=0 and comm.rank !=1 :
				A_prime = np.empty((len(X_all_ind), 6))
				noadj_vals = np.empty((len(X_all_ind), 6))
				rent_vals = np.empty((len(X_all_ind), 6))
				adj_vals = np.empty((len(X_all_ind), 6))


			# Recieve adj policies on rank 0 from rank 1
			if comm.rank == 0:
				C_adj = np.empty((len(DB), grid_size_W,
					   grid_size_alpha,
					   grid_size_beta,
					   len(Pi),
					   grid_size_DC,
					   grid_size_Q,
					   grid_size_M,
					   grid_size_A))
				H_adj = np.copy(C_adj)
				Aprime_adj = np.copy(H_adj)
				comm.Recv(adj_vals, source =1, tag= 4)
				comm.Recv(C_adj, source =1, tag= 5)
				comm.Recv(H_adj, source =1, tag= 6)
				comm.Recv(Aprime_adj, source =1, tag= 7)
				if verbose == True:
					print("Solved eval_policy_W of age {} in {} seconds"\
					.format(Age, time.time() - start))
			
			# Step 4: Evaulate unconditioned RHS of Euler equations rank 0 and 1
			start = time.time()
			#if comm.rank == 0 or comm.rank == 1:
			B_funcs = gen_RHS_UC_func(comm, t,
					  VF,
					  UC_prime_H,
					  A_prime, 
					  noadj_vals,
					  adj_vals,
					  rent_vals)
						 
			if comm.rank == 0:
				(UC_prime_B, UC_prime_H_B, UC_prime_M_B, VF_B, Xi, UF_B, zeta) =  B_funcs

				if verbose == True:
					print("Solved gen_RHS_UC_func of age {} in {} seconds".
					format(Age, time.time() - start))

				# Step 5: Condition out discrete choice and preference shocks
				start = time.time()
				UC_prime_DCC, UC_prime_H_DCC,\
					UC_prime_M_DCC, VF_DCC,\
					prob_v, prob_pi,Xi_copi_temp = UC_cond_DC(UC_prime_B,
														 UC_prime_H_B,
														 UC_prime_M_B, VF_B,\
														 Xi)
				if verbose == True:
					print("Solved UC_cond_DC of age {} in {} seconds".
						  format(Age, time.time() - start))

				# Step 6: Condition out DC return, house price and mortgage 
				# (iid) shocks
				start = time.time()
				UC_prime, UC_prime_H, UC_prime_M, VF\
					= UC_cond_all(t,
								  UC_prime_DCC,
								  UC_prime_H_DCC,
								  UC_prime_M_DCC, VF_DCC)
				if verbose == True:
					print("Solved UC_cond_all of age {} in {} seconds".
						  format(Age, time.time() - start))
					print("Iteration time was {}".format(time.time() - start2))

				# Step 7: Save policy functions to scratch
				start = time.time()
				if t == tzero:
					policy_VF = VF.reshape((len(DB),
						 grid_size_W,
						 grid_size_alpha,
						 grid_size_beta,
						 len(Pi),
						 grid_size_A,
						 grid_size_DC,
						 grid_size_H,
						 grid_size_Q,
						 grid_size_M))
					np.savez_compressed("{}/age_{}_acc_{}_id_{}_pols".
										format(pol_path_id, t, acc_ind[0], ID),
										C_adj = C_adj,
										H_adj = H_adj,
										Aprime_adj = Aprime_adj,
										C_noadj = C_noadj,
										etas_noadj = np.log(
											np.abs((etas_noadj))).astype(np.float32),
										Aprime_noadj = Aprime_noadj,
										zeta = np.log(zeta).astype(np.float32),
										H_rent = H_rent,
										prob_v = prob_v,
										prob_pi = prob_pi,
										policy_VF = policy_VF)
					if verbose == True:
						print("Saved policies in {} seconds"
							  .format(-time.time() + time.time()))
					del C_adj, H_adj, Aprime_adj, C_noadj,etas_noadj,Aprime_noadj,zeta,H_rent,prob_v,prob_pi
					gc.collect()
				else:
					np.savez_compressed("{}/age_{}_acc_{}_id_{}_pols".
										format(pol_path_id, t, acc_ind[0], ID),
										C_adj = C_adj,
										H_adj = H_adj,
										Aprime_adj = Aprime_adj,
										C_noadj = C_noadj,
										etas_noadj = np.log(np.abs((etas_noadj))).astype(np.float32),
										Aprime_noadj = Aprime_noadj,
										zeta = np.log(zeta).astype(np.float32),
										H_rent = H_rent,
										prob_v = prob_v,
										prob_pi = prob_pi)
					if verbose == True:
						print("Saved policies in {} seconds"
							  .format(-time.time() + time.time()))
					
					del C_adj, H_adj, Aprime_adj, C_noadj,etas_noadj,Aprime_noadj,zeta,H_rent,prob_v,prob_pi
					gc.collect()

			# On all ranks other than 0, reset values of t+1, t conditoned 
			# value functions and marginal utilties 
			if comm.rank!=0:
				UC_prime = np.empty(np.shape(UC_prime),dtype=np.float64)
				UC_prime_H = np.empty(np.shape(UC_prime_H),dtype=np.float64)
				UC_prime_M = np.empty(np.shape(UC_prime_M),dtype=np.float64)
				VF = np.empty(np.shape(VF),dtype=np.float64)
				UF = np.empty(np.shape(UF),dtype=np.float64)

			# From rank 0, broadcast value functions and marginal utilties 
			# conditioned on time t 
			comm.Bcast([UC_prime,MPI.DOUBLE], root = 0)
			comm.Bcast([UC_prime_H,MPI.DOUBLE],root = 0)
			comm.Bcast([UC_prime_M,MPI.DOUBLE], root = 0)
			comm.Bcast([VF,MPI.DOUBLE], root = 0)
			comm.Barrier()
		
		if verbose == True: 
			print("Solved lifecycle model in {} seconds".format(
				time.time() - start1))
		return ID
	return solve_LC_model


def generate_worker_pols(og,
						 world_comm,
						 comm,
						 load_retiree = 1,
						 scr_path ='/scratch/pv33/ls_model_temp2/',
						 gen_newpoints = False):

	if comm.rank == 0:
		from solve_policies.retiree_solver import retiree_func_factory
		gen_R_pol = retiree_func_factory(og)
	else:
		gen_R_pol = None

	solve_LC_model = worker_solver_factory(og, world_comm, comm, gen_R_pol,
										   scr_path = scr_path,
										   gen_newpoints = gen_newpoints,
										   verbose = True)
	del og
	og = {}
	gc.collect()
	policies = solve_LC_model(comm, load_retiree)

	return policies
