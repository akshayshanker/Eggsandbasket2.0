import numpy as np 
from numba import njit, prange
import subprocess
import csv
import yaml
from interpolation.splines import extrap_options as xto
from interpolation.splines import eval_linear

@njit(cache = True)
def eval_linear_c(grid, func, point, xto):
	return eval_linear(grid, func, point, xto)

def get_jobfs_path():
	batcmd = "cd $PBS_JOBFS;pwd"
	result = subprocess.check_output(batcmd, shell=True, text = True).strip("\n")

	return result 

def read_settings(settings_path):
	with open(settings_path + "/settings.yml", "r") as stream:
		eggbasket_config = yaml.safe_load(stream)

	param_random_bounds = {}
	with open(settings_path + '/random_param_bounds.csv', newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],\
				row['UB']])
	return eggbasket_config, param_random_bounds


def gen_policyout_arrays(og):


	# grid parameters 
	DC_max                          = og.parameters.DC_max
	C_min, C_max                    = og.parameters.C_min, og.parameters.C_max 
	Q_max,A_min                     = og.parameters.Q_max, og.parameters.A_min
	H_min, H_max                    = og.parameters.H_min, og.parameters.H_max
	A_max_W                         = og.parameters.A_max_W
	R, tzero                        = og.parameters.R, og.parameters.tzero
	DB = og.grid1d.DB

	grid_size_A, grid_size_DC, grid_size_H,\
	grid_size_Q, grid_size_M, grid_size_C = og.parameters.grid_size_A,\
									  og.parameters.grid_size_DC,\
									  og.parameters.grid_size_H,\
									  og.parameters.grid_size_Q,\
									  og.parameters.grid_size_M,\
									  og.parameters.grid_size_C
	grid_size_HS = og.parameters.grid_size_HS

	grid_size_W, grid_size_alpha, grid_size_beta\
									=  int(og.parameters.grid_size_W),\
									int(og.parameters.grid_size_alpha),\
									 int(og.parameters.grid_size_beta)


	# smaller, "fundamental" grids 
	A_DC,H,A,M,Q                  = og.grid1d.A_DC, og.grid1d.H,og.grid1d.A, og.grid1d.M, og.grid1d.Q
	W_W                             = og.grid1d.W_W
	V, Pi, DB                       = og.grid1d.V, og.grid1d.Pi, og.grid1d.DB

	
	all_state_shape                 = (1,grid_size_W,\
										 grid_size_alpha,\
										 grid_size_beta,\
										 len(Pi),\
										 grid_size_A,\
										 grid_size_DC,\
										 grid_size_H,\
										 grid_size_Q,\
										 grid_size_M)

	all_state_shape_hat             = (1,grid_size_W,\
										grid_size_alpha,\
										grid_size_beta,\
										len(V),\
										len(Pi),\
										grid_size_A,\
										grid_size_DC,\
										grid_size_H,\
										grid_size_Q,\
										grid_size_M)

	v_func_shape                    = (int(1*grid_size_W\
										 *grid_size_alpha\
										 *grid_size_beta\
										 *len(Pi)*grid_size_Q),\
										 grid_size_A,\
										 grid_size_DC,\
										 grid_size_H,\
										 grid_size_M)

	all_state_A_last_shape          = (1,grid_size_W,\
										 grid_size_alpha,\
										 grid_size_beta,\
										 len(Pi),\
										 grid_size_DC,\
										 grid_size_H,\
										 grid_size_Q,\
										 grid_size_M,\
										 grid_size_A)


	policy_shape_nadj = (int(R)-int(tzero), 1, grid_size_W, grid_size_alpha,\
							grid_size_beta, len(Pi),\
							grid_size_A,grid_size_DC,\
							grid_size_H, grid_size_Q,\
							grid_size_M)

	policy_shape_adj = (int(R)-int(tzero), 1, grid_size_W, grid_size_alpha,\
							grid_size_beta, len(Pi),\
							grid_size_DC,\
							grid_size_Q,\
							grid_size_M,\
							grid_size_A)

	policy_shape_rent = (int(R)-int(tzero),1,
								grid_size_W,
								grid_size_alpha,
								grid_size_beta,
								len(Pi),
								grid_size_A,
								grid_size_DC,
								grid_size_Q,
								)


	prob_v_shape = (int(1), grid_size_W,
						grid_size_alpha,
						grid_size_beta,
						grid_size_A,
						grid_size_DC,
						grid_size_H,
						grid_size_Q,
						grid_size_M,
						len(V))

	prob_pi_shape = (int(1), grid_size_W,
						  grid_size_alpha,
						  grid_size_beta,
						  len(V),
						  grid_size_A,
						  grid_size_DC,
						  grid_size_H,
						  grid_size_Q,
						  grid_size_M,
						  len(Pi))
	return all_state_shape,all_state_shape_hat, v_func_shape,\
			all_state_A_last_shape, policy_shape_nadj, policy_shape_adj,\
			policy_shape_rent, prob_v_shape, prob_pi_shape

@njit(cache = True)
def sim_markov(P, P_stat, U):
	""" Simulate markov chain length
		len(U) with probability matrix
		P
	"""

	#index of states

	states = np.arange(len(P_stat))

	# initial draw from stationary distribution

	index_series = np.zeros(len(U))

	index_series[0]= states[np.searchsorted(np.cumsum(P_stat), U[0])]

	for i in range(len(U)):
		state_i = int(index_series[i])
		trans_prob_i = P[state_i]
		index_series[i+1] = states[np.searchsorted(np.cumsum(trans_prob_i), U[i+1])]

	return index_series

@njit(cache = True)
def d0(v1,v2):
	""" eval inner product of two arrays
	""" 

	out = 0
	for k in range(len(v1)):
		out += v1[k] * v2[k]
	return out

@njit(cache = True)
def interp_as(xp,yp,x, extrap = True):

	"""Function  interpolates 1D
	with linear extraplolation 

	Parameters
	----------
	xp : 1D array
		  points of x values
	yp : 1D array
		  points of y values
	x  : 1D array
		  points to interpolate 

	Returns
	-------
	evals: 1D array  
			y values at x 

	"""

	evals = np.zeros(len(x))
	if extrap == True and len(xp)>1:
		for i in range(len(x)):
			if x[i]< xp[0]:
				if (xp[1]-xp[0])!=0:
					evals[i]= yp[0]+(x[i]-xp[0])*(yp[1]-yp[0])\
						/(xp[1]-xp[0])
				else:
					evals[i] = yp[0]

			elif x[i] > xp[-1]:
				if (xp[-1]-xp[-2])!=0:
					evals[i]= yp[-1]+(x[i]-xp[-1])*(yp[-1]-yp[-2])\
						/(xp[-1]-xp[-2])
				else:
					evals[i] = yp[-1]
			else:
				evals[i]= np.interp(x[i],xp,yp)
	else:
		evals = np.interp(x,xp,yp)
	return evals


@njit(cache = True)
def dot_py(A,B):
	m, n = A.shape
	p = B.shape[1]

	C = np.zeros((m,p))

	for i in range(0,m):
		for j in range(0,p):
			for k in range(0,n):
				C[i,j] += A[i,k]*B[k,j] 
	return C

@njit(cache = True)
def _my_einsum_row(my_A,my_B):
	my_out = np.empty(len(my_A))
	for i in range(len(my_A)):
		x = 0
		for j in range(len(my_A[0])):
			x+= my_A[i,j]*my_B[i,j]
		
		my_out[i] = x 
	
	return my_out 

def einsum_row(A,B,comm):
	if comm.rank == 0:
		out = np.empty(len(A))
		A_split = np.split(A, comm.size, axis = 0)
		B_split = np.split(B, comm.size, axis = 0)
	else:
		out = None
		A_split = None
		B_split = None

	my_A = comm.scatter(A_split, root = 0)
	my_B = comm.scatter(B_split, root = 0)

	my_out = _my_einsum_row(my_A, my_B)

	sendcounts = np.array(comm.gather(len(my_out), 0))

	comm.Gatherv(np.ascontiguousarray(my_out), recvbuf=(out, sendcounts), root =0)

	if comm.rank == 0:
		return out
	else:
		return None

def gen_reshape_funcs(og):

	grid_size_A, grid_size_DC, grid_size_H,\
	grid_size_Q, grid_size_M, grid_size_C = og.parameters.grid_size_A,\
									  og.parameters.grid_size_DC,\
									  og.parameters.grid_size_H,\
									  og.parameters.grid_size_Q,\
									  og.parameters.grid_size_M,\
									  og.parameters.grid_size_C
	grid_size_HS = og.parameters.grid_size_HS

	grid_size_W, grid_size_alpha, grid_size_beta = int(og.parameters.grid_size_W),\
													int(og.parameters.grid_size_alpha),\
													int(og.parameters.grid_size_beta)
	
	DB, V, Pi, X_W_bar_hdjex_ind                 = og.grid1d.DB, og.grid1d.V, og.grid1d.Pi, og.BigAssGrids.X_W_bar_hdjex_ind_f().astype(np.int32)

	EBA_P2                                       = og.cart_grids.EBA_P2

	all_state_shape,all_state_shape_hat, v_func_shape,\
	all_state_A_last_shape, policy_shape_nadj, policy_shape_adj, policy_shape_rent,\
	prob_v_shape, prob_pi_shape = gen_policyout_arrays(og)  
								   
	def reshape_out_pi(UC_in):
		"""
		Reshapes 1D (flat 10D) array to 2D array with Pi values
		as cols and cart product of all other states as rows
		"""

		UC_in = UC_in.reshape((1, grid_size_W,\
											 grid_size_alpha,\
											 grid_size_beta,\
											 len(V),\
											 len(Pi),\
											 grid_size_A,\
											 grid_size_DC,\
											 grid_size_H,\
											 grid_size_Q,\
											 grid_size_M,\
											 ))
		

		UC_in         = np.transpose(UC_in, (0,1,2,3,4,6,7,8,9,10,5) )

		UC_in          = UC_in.reshape(int(1*grid_size_W*\
											 grid_size_alpha*\
											 grid_size_beta*\
											 len(V)*\
											 grid_size_A*\
											 grid_size_DC*\
											 grid_size_H*\
											 grid_size_Q*\
											 grid_size_M),\
											 len(Pi))

		return UC_in
	
	def reshape_out_V(UC_in):
		"""
		Reshapes 1D (flat 10D) array to 2D array with V values
		as coloumns and all other states as rows
		"""

		UC_in = UC_in.reshape((1, grid_size_W,\
											 grid_size_alpha,\
											 grid_size_beta,\
											 len(V),\
											 grid_size_A,\
											 grid_size_DC,\
											 grid_size_H,\
											 grid_size_Q,\
											 grid_size_M,\
											 ))
		
		UC_in         = np.transpose(UC_in, (0,1,2,3,5,6,7,8,9,4) )

		UC_in          = UC_in.reshape(int(1*grid_size_W*\
											 grid_size_alpha*\
											 grid_size_beta*\
											 grid_size_A*\
											 grid_size_DC*\
											 grid_size_H*\
											 grid_size_Q*\
											 grid_size_M),\
											 len(V)
											 )
		return UC_in

	def reshape_X_bar(UC_in):
		"""
		Reshapes 1D array (without V and Pi) indexed 
		by all 9 states other than V and Pi to 9D array 

		Conditions arrays defined on time t E, Alpha and Beta 
		to time t-1  E, Alpha and Beta 
		"""

		UC_in = UC_in.reshape((1, grid_size_W,\
											 grid_size_alpha,\
											 grid_size_beta,\
											 grid_size_A,\
											 grid_size_DC,\
											 grid_size_H,\
											 grid_size_Q,\
											 grid_size_M,\
											 ))

		UC_in = UC_in.transpose((1,2,3,0,4,5,6,7,8))

		UC_in  = UC_in.reshape((int(grid_size_W*\
											 grid_size_alpha*\
											 grid_size_beta),\
											 int(1*grid_size_A*\
											 grid_size_DC*\
											 grid_size_H*\
											 grid_size_Q*grid_size_M)
											 ))

		U_condE  = dot_py(EBA_P2, UC_in)

		U_condE = U_condE.reshape((grid_size_W,\
											 grid_size_alpha,\
											 grid_size_beta,\
											 1,grid_size_A,\
											 grid_size_DC,\
											 grid_size_H,\
											 grid_size_Q,\
											 grid_size_M
											 ))

		U_out = U_condE.transpose((3,0,1,2,4,5,6,7,8))

		return U_out

	@njit
	def reshape_make_Apfunc_last(UC):

		UC_R1     = UC.reshape((1, grid_size_W,\
						 grid_size_alpha, grid_size_beta,\
						 len(Pi), grid_size_A,\
						 grid_size_DC,\
						 grid_size_H, grid_size_Q,\
						 grid_size_M))

		UC_R1        = np.transpose(UC_R1,\
									 (0,1,2,3,4,6,7,8,9,5))

		UC_R         = np.copy(UC_R1).\
							reshape((int(len(X_all_ind_vals)\
							/grid_size_A), int(grid_size_A)))
		return UC_R
		
	def reshape_make_h_last(A_prime):
		A_prime_adj_reshape1     = A_prime.reshape((1, grid_size_W,\
									grid_size_alpha, grid_size_beta,\
									len(Pi), grid_size_DC, grid_size_H,\
									grid_size_Q))
		A_prime_adj_reshape2     = A_prime_adj_reshape1.\
									transpose((0,1,2,3,4,5,7,6))

		A_prime_adj_reshape      = A_prime_adj_reshape2.\
									reshape((int(len(X_W_bar_hdjex_ind)\
										/grid_size_H), int(grid_size_H)))

		return A_prime_adj_reshape

	def reshape_adj_RHS(adj_pols):

			# Recall the adjustment policies are ordered as:
			#((len(DB)(0), grid_size_W(1),\
			#              grid_size_alpha(2),\
			#              grid_size_beta(3),\
			#              len(Pi)(4),\
			#              grid_size_DC(5),\
			#              grid_size_Q(6),grid_size_M(7),\
			#              grid_size_A(8)))
			# transpose so we have 
			# DBxExAlphaxBetaxPixQxMxWealthxA_DC
		adj_pols            = adj_pols.transpose((0,1,2,3,4,6,7,8,5))
		adj_pols   = adj_pols.reshape((int(1*grid_size_W\
								*grid_size_alpha\
								*grid_size_beta*len(Pi)\
								*grid_size_Q*grid_size_M),\
								 grid_size_A, grid_size_DC ))

		# recall the renter polices have shape
		# DB(0), E(1), Alpha(2), beta(3), Pi(4), A_DC(5), Q(6),Wealth(7)
		# re-order so we have 
		# DB(0)xE(1)xAlpha(2)xBeta(3)xPi(4)xQ(6)xWealth(7)xA_DC(5)
		
		return adj_pols 

	def reshape_rent_RHS(rent_pols):

		rent_pols = rent_pols.transpose((0,1,2,3,4,6,7,5))
		rent_pols = rent_pols.reshape((int(1*grid_size_W*grid_size_alpha*
									 grid_size_beta*len(Pi)*\
									 grid_size_Q),\
									 grid_size_A, grid_size_DC ))

		return rent_pols


	def reshape_nadj_RHS(nadj_pols1):

		# noadj funcs are ordered as
		# DB(0)xE(1)xAlpha(2)xBeta(3)xPi(4)xA(5)xA_DC(6)xH(7)xQ(8)xM(9)
		# transpose so we have 
		# DB(0)xE(1)xAlpha(2)xBeta(3)xPi(4)xH(7)xQ(8)xM(9)xA(5)xA_DC(6)
		nadj_pols1 = nadj_pols1.transpose((0,1,2,3,4,7,8,9,5,6))
		nadj_pols1= nadj_pols1.reshape((int(1*grid_size_W*grid_size_alpha*
								 grid_size_beta*len(Pi)*grid_size_H*\
								 grid_size_Q*grid_size_M),\
								 grid_size_A, grid_size_DC ))
	
		return nadj_pols1

	def reshape_vfunc_points(a_prime_norent_vals, adcprime, h_prime_norent_vals, m_prime):

		# points are in shape: 
		# |DB(0)xE(1)XA(2)xB(3)xPi(4)xV(5)xA(6)xDC(7)xH(8)xQ(9)xM(10)|x points(11)
		# first re-order to: 
		# DB(0)xE(1)xA(2)xB(3)xPi(4)xQ(9)xV(5)xA(6)xDC(7)xH(8)xM(10)x points(11)
		# then reshape back to:
		# |DB(0)xE(1)xA(2)xB(3)xPi(4)xQ(9)|x|V(5)xA(6)xDC(7)xH(8)xM(10)|x points(11)
		points = np.column_stack(
			(a_prime_norent_vals, adcprime, h_prime_norent_vals, m_prime))

		points = points.reshape((1,grid_size_W,\
									grid_size_alpha,\
									grid_size_beta,\
									len(Pi),\
									len(V),\
									grid_size_A,
									grid_size_DC,\
									grid_size_H,\
									grid_size_Q,\
									grid_size_M,4))

		points = points.transpose((0,1,2,3,4,9,5,6,7,8,10,11))
		points = points.reshape((int(1*grid_size_W*grid_size_alpha*grid_size_beta*len(Pi)*grid_size_Q),\
										int(len(V)*grid_size_A*grid_size_DC*grid_size_H*grid_size_M),4))


		return points

	def reshape_RHS_UFB(UF_FUNC):

		UF_FUNC = UF_FUNC.reshape(all_state_shape_hat)
		UF_FUNC = UF_FUNC.transpose((0,1,2,3,4,9,5,6,7,8,10))
		UF_FUNC = UF_FUNC.reshape((int(1*grid_size_W\
														*grid_size_alpha\
														*grid_size_beta\
														*len(Pi)*grid_size_Q),\
														int(grid_size_A*\
															grid_size_DC*\
															len(V)*
															grid_size_H*\
															grid_size_M)))

		return UF_FUNC

	def reshape_RHS_Vfunc_rev(Xifunc):
		""" Reshapes  policies interpolated
		from the t+1 value function back to the order
		of X_all_ind"""

		Xifunc  = Xifunc.reshape((1,\
												grid_size_W,\
												grid_size_alpha,\
												grid_size_beta,\
												len(Pi),grid_size_Q,\
												len(V),\
												grid_size_A,\
												grid_size_DC,\
												grid_size_H,\
												grid_size_M))

		# change order so Pi, V are consecutive and Q, M are consec
		Xifunc = Xifunc.transpose((0,1,2,3,6,4,7,8,9,5,10))

		# unravel array 
		Xifunc = Xifunc.reshape(int(1*grid_size_W\
												*grid_size_alpha\
												*grid_size_beta\
												*len(Pi)*grid_size_Q*\
												grid_size_A*\
												grid_size_DC*\
												len(V)*\
												grid_size_H*\
												grid_size_M))
		return Xifunc

	def _rehape_adj_post_interp(policy_1):

		policy = policy_1.reshape((len(DB), grid_size_W,
									   grid_size_alpha,
									   grid_size_beta,
									   len(Pi),
									   grid_size_DC,
									   grid_size_Q,
									   grid_size_A))

		# Repeat
		policy = np.stack((policy,) * grid_size_M, axis=-1)

		policy = policy.transpose((0,1,2,3,4,5,6,8,7))

		return policy

	return     reshape_out_pi, reshape_out_V, reshape_X_bar,\
				reshape_make_Apfunc_last, reshape_make_h_last,\
				reshape_vfunc_points,reshape_nadj_RHS,reshape_rent_RHS,\
				reshape_adj_RHS, reshape_RHS_Vfunc_rev,reshape_RHS_UFB,\
				reshape_vfunc_points, _rehape_adj_post_interp

