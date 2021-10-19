
"""
This module contains the LifeCycleModel class
 
Classes: LifeCycleModel 
				 Class with parameters, grids and functions 
				 for a LS model

Example use:

# Create housing model 
og = HousingModel(config, account_type)

# Solve worker policies  
from solve_policies.worker_solver import generate_worker_pols

policy = generate_worker_pols(og)
		
"""

# Import packages

from util.randparam import rand_p_generator
from util.ls_model_functions import lsmodel_function_factory
from util.ls_model_functions_CD import lsmodel_function_factory as lsmodel_function_factory_CD
import numpy as np
from numba import njit, prange, guvectorize, jit
import time
import random
import string
import dill as pickle
from sklearn.utils.extmath import cartesian
from quantecon import tauchen
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
import copy
from pathlib import Path


import warnings
warnings.filterwarnings('ignore')


class LifeCycleModel:

		"""
		An instance of this class is a LS model with
		parameters, functions, asset grids survival probabilities 
		and shock processes. 

		Parameters 
		----------
		config : dict
						 dictionary with parameters and settings 
		acc_ind : 1D array
							Account type (0: DB, 1: DC)


		Attributes
		----------
		parameters : 
										ModelParameters instance with all model parameters 
		functions : 
										ModelFunctions instance with model functions 
		grid1d : 
										Model1DGrids instance with small 1D grids
		st_grid : 
										ModelStochProcGrids instance with shock grids 
		big_grids :
										BigAssGrids instance with all large cartesian grids

		Todo
		----
						Check the scaling factors in the low and high risk returns
						processes are consistent with paper; need to confirm 
						whether sigma_DC and sigma_DB is variance or SD?

						Check and revise calculation of r_m_prime in grids_generate and helper
						funcs. Apply DRY to calculation r_m_prime

						Change paramter h to another name

						change paramters r_h to another name 

		Completed checkes
		-----------------

		"""

		def __init__(self,
								 config,        # Settings
								 acc_ind,       # Grid of accound ind
								 param_id,
								 mod_name
								 ):
				self.parameters = self.ModelParameters(config)
				self.functions = self.ModelFunctions(config)
				self.functions_CD = self.ModelFunctions_CD(config)
				self.grid1d = self.Model1DGrids(acc_ind, config, self.parameters)
				self.interp_grid = self.ModelInterpGrids(self.parameters)
				self.st_grid = self.ModelStochProcGrids(config)
				self.cart_grids = self.ModelCartGrids(self.grid1d,
																							self.st_grid,
																							self.parameters)
				self.BigAssGrids = self.BigAssGrids(
						self.grid1d, self.st_grid, self.parameters)

				self.acc_ind = acc_ind
				self.mod_name = mod_name

				# self.parameters.phi_r = eval_rent_share(self.st_grid.Q_shocks_r,
 #                                        self.parameters.r,
 #                                         self.parameters.r_H,
 #                                         self.parameters.r_l,
 #                                         self.parameters.beta_m)

				self.ID = param_id

		class ModelParameters:

				""" All model parameters and grid sizes
				"""

				def __init__(self, config):
						parameters = config['parameters']
						lambdas = np.array(config['lambdas'])
						normalisation = np.array(config['normalisation'])

						self.__dict__.update(config['parameters'])


						self.s = np.array(config['survival'])
						self.T = int(self.T)
						# overide pre-set welath and mortgage max
						#self.M_max = self.H_max*self.Q_max*(1-self.phi_c)
						self.A_max_WW = self.H_max + self.A_max_W
						self.A_max_WE = self.H_max + self.A_max_R


						# set grid sizes as int
						self.grid_size_A, self.grid_size_DC, \
								self.grid_size_H, self.grid_size_Q, self.grid_size_M, \
								self.grid_size_DC = int(parameters['grid_size_A']),\
								int(parameters['grid_size_DC']),\
								int(parameters['grid_size_H']),\
								int(parameters['grid_size_Q']),\
								int(parameters['grid_size_M']),\
								int(parameters['grid_size_C'])
						self.grid_size_Q_s = int(parameters['grid_size_Q_s'])
						self.grid_size_DCR = int(parameters['grid_size_DCR'])
						self.grid_size_DC = int(parameters['grid_size_DC'])
						self.grid_size_W = int(parameters['grid_size_W'])
						self.grid_size_HS = int(parameters['grid_size_HS'])

		class ModelFunctions:

				""" Functions paramaterized for LS model 
				"""

				def __init__(self, config):
						parameters = config['parameters']
						lambdas = np.array(config['lambdas'])
						normalisation = np.array(config['normalisation'])

						# generate the functions for the housing model
						self.u, self.uc, self.uh, self.b, self.b_prime, self.y,\
								self.yvec, self.DB_benefit, self.adj_p, self.adj_v,\
								self.adj_pi, self.uc_inv, self.uh_inv, self.amort_rate,\
								self.u_vec, self.uc_vec, self.housing_ser,\
								self.ch_ser, self.ucnz, self.ces_c1 = lsmodel_function_factory(parameters,
																					 lambdas,
																					 normalisation)
		class ModelFunctions_CD:

			""" Functions paramaterized for LS model 
			"""

			def __init__(self, config):
					parameters = config['parameters']
					lambdas = np.array(config['lambdas'])
					normalisation = np.array(config['normalisation'])

					# generate the functions for the housing model
					self.u, self.uc, self.uh, self.b, self.b_prime, self.y,\
							self.yvec, self.DB_benefit, self.adj_p, self.adj_v,\
							self.adj_pi, self.uc_inv, self.uh_inv, self.amort_rate,\
							self.u_vec, self.uc_vec, self.housing_ser,\
							self.ch_ser, self.ucnz = lsmodel_function_factory_CD(parameters,
																				 lambdas,
																				 normalisation)
		class Model1DGrids:

				""" 
				1D grids of all non-shock states 

				Attributes
				----------

				A : 1D array
						 liquid assets workers
				A_DC : 1D array
							pension asset  grid
				A_R : 1D array
																 retired liquid asset  grid 
				H : 1D array
												housing grid
				Q : 1D array
												house price  grid 
				M : 1D array
												Mortgage grid 
				W_R : 1D array
																liquid cash at hand wealth for retirees 
				W_W : 1D array
																liquid cash at hand wealth for workers 
				H_R : 1D array
																housing services grid 
				V : 1D array
												 voluntary contribution grid
				Pi : 1D array
												 Risk share grid 
				DB : 1D array
												 Account type grid 

				"""

				def __init__(self, acc_ind, config, parameters):

						param = parameters

						# generate 1D grids
						self.W_R = np.linspace(param.A_min, param.A_max_WE,
																	 param.grid_size_A)
						self.W_W = np.linspace(param.A_min, param.A_max_WW,
																	 param.grid_size_A)
						self.A_R = np.linspace(param.A_min, param.A_max_R,
																	 param.grid_size_A)
						self.A = np.linspace(param.A_min_W, param.A_max_W,
																 param.grid_size_A)

						self.A_DC = np.linspace(param.DC_min, param.DC_max,
																		param.grid_size_DC)
						self.H = np.linspace(param.H_min, param.H_max,
																 param.grid_size_H)
						self.Q = np.linspace(param.Q_min, param.Q_max,
																 param.grid_size_Q)
						self.M = np.linspace(param.M_min, param.M_max,
																 param.grid_size_M)
						self.H_R = np.linspace(param.H_min, param.HS_max,
																	 int(param.grid_size_HS))

						self.DB = acc_ind
						self.V = np.array(config['vol_cont_points'])
						self.Pi = np.array(config['risk_share_points'])

		class ModelInterpGrids:
				def __init__(self, parameters):
						param = parameters

						# Cartesian grids
						self.W_Q_R\
								= UCGrid((param.A_min, param.A_max_WE, param.grid_size_A),
												 (param.Q_min, param.Q_max, param.grid_size_Q))

						self.X_cont_R\
								= UCGrid((param.A_min, param.A_max_R, param.grid_size_A),
												 (param.H_min, param.H_max, param.grid_size_H),
												 (param.Q_min, param.Q_max, param.grid_size_Q),
												 (param.M_min, param.M_max, param.grid_size_M))
						self.X_cont_RC\
								= UCGrid((param.A_min, param.A_max_R, param.grid_size_A),
												 (param.H_min, param.H_max, param.grid_size_H),
												 (param.Q_min, param.Q_max, param.grid_size_Q))
						self.X_QH_R\
								= UCGrid((param.Q_min, param.Q_max, param.grid_size_Q),
												 (param.A_min, param.A_max_WE, param.grid_size_A))
						self.X_QH_W\
								= UCGrid((param.A_min_W, param.A_max_WW, param.grid_size_A),
												 (param.DC_min, param.DC_max, param.grid_size_DC))

						self.X_QH_WRTS\
								= UCGrid((param.A_min_W, param.A_max_WW, param.grid_size_A),
												 (param.DC_min, param.DC_max, param.grid_size_DC),
												 (param.Q_min, param.Q_max, param.grid_size_Q))
						self.X_DCQ_W\
								= UCGrid((param.DC_min, param.DC_max, param.grid_size_DC),
												 (param.Q_min, param.Q_max, param.grid_size_Q),
												 (param.M_min, param.M_max, param.grid_size_M))
						self.X_QH_W_TS\
								= UCGrid((param.DC_min, param.DC_max, param.grid_size_DC),
												 (param.Q_min, param.Q_max, param.grid_size_Q),
												 (param.M_min, param.M_max, param.grid_size_M),
												 (param.A_min_W, param.A_max_WW, param.grid_size_A))
						self.X_cont_WAM2\
								= UCGrid((param.A_min, param.A_max_W, param.grid_size_A),
												 (param.H_min, param.H_max, param.grid_size_H),
												 (param.M_min, param.M_max, param.grid_size_M))
						self.X_cont_WAM\
								= UCGrid((param.A_min, param.A_max_W, param.grid_size_A),
												 (param.M_min, param.M_max, param.grid_size_M))
						self.X_cont_AH\
								= UCGrid((param.A_min, param.A_max_W, param.grid_size_A),
												 (param.H_min, param.H_max, param.grid_size_H))
						self.X_cont_W\
								= UCGrid((param.A_min, param.A_max_W, param.grid_size_A),
												 (param.DC_min, param.DC_max, param.grid_size_DC),
												 (param.H_min, param.H_max, param.grid_size_H),
												 (param.Q_min, param.Q_max, param.grid_size_Q),
												 (param.M_min, param.M_max, param.grid_size_M))
						self.X_cont_W_bar\
								= UCGrid((param.A_min, param.A_max_W, param.grid_size_A),
												 (param.DC_min, param.DC_max, param.grid_size_DC),
												 (param.H_min, param.H_max, param.grid_size_H),
												 (param.M_min, param.M_max, param.grid_size_M))
						self.X_cont_W_hat\
								= UCGrid((param.A_min, param.A_max_W, param.grid_size_A),
												 (param.DC_min, param.DC_max, param.grid_size_DC))

						self.X_W_contgp = nodes(self.X_cont_W)
						self.X_RC_contgp = nodes(self.X_cont_RC)
						self.X_R_contgp = nodes(self.X_cont_R)

		class ModelStochProcGrids:

				"""
				Grids for the exogenous stochastic shocks

				Attributes
				----------

				E : 1D array
												income shock grid
				alpha_hat : 1D array
																												ln(alpha) process shock value grid 
				P_alpha   : 2D array
																												 transition matrix for alpha shock
				beta_hat : 1D array
																												ln(beta) beta shock value grid
				P_beta : 2D array
																								transition matrix for beta shock
				P_stat : 1D array
																								stationary dist of wage shock
				P_E : 2D array
																				wage shock transition probabilities
				X_rl : 1D array
																				low risk DC asset return grid
				P_rl : 1D array
																				shock probs for `X_rl'
				X_rh : 1D array
																				high risk DC asset return grid
				P_rh : 1D array
																				shock probs for `X_rh'
				X_r :  2D array
																				Cartesian product of DC high and low return shock vals
				P_r : 1D arrat
																				Joint prob distribution of X_r
				Q_shocks_r : 1D array
																												 House price shock s
				Q_shocks_P : 2D array
																																Transition prob. matrix for house price shock

				"""

				def __init__(self, config):
						parameters = config['parameters']

						# Labor shock process
						self.labour_mc = tauchen(parameters['phi_w'],
																		 parameters['sigma_w'],
																		 n=int(parameters['grid_size_W']))
						self.E = self.labour_mc.state_values
						self.P_E = self.labour_mc.P
						self.P_stat = self.labour_mc.stationary_distributions[0]

						#  beta and alpha processes
						#  Recall that beta_hat are values of ln beta_{t} - ln beta_\bar
						#  alpha_hat are values of ln beta_{t} - ln beta_\bar

						lnr_beta_bar = np.log(1/parameters['beta_bar'] - 1)
						y_bar = lnr_beta_bar*(1-parameters['rho_beta'])
						self.lnr_beta_mc = tauchen(parameters['rho_beta'],
																			 parameters['sigma_beta'],
																			 b=y_bar,
																			 n=int(parameters['grid_size_beta']))
						self.beta_hat = 1/(1+np.exp(self.lnr_beta_mc.state_values))

						self.P_beta = self.lnr_beta_mc.P
						self.beta_stat = self.lnr_beta_mc.stationary_distributions[0]
						
						lnr_alpha_bar = np.log(1/parameters['alpha_bar'] - 1)
						y_bar_alpha =  lnr_alpha_bar*(1-parameters['rho_alpha'])

						self.lnr_alpha_mc = tauchen(parameters['rho_alpha'],
																		parameters['sigma_alpha'],
																		b = y_bar_alpha,
																		n=int(parameters['grid_size_alpha']))
						
						self.alpha_hat = 1/(1+ np.exp(self.lnr_alpha_mc.state_values))

						self.P_alpha = self.lnr_alpha_mc.P
						self.alpha_stat = self.lnr_alpha_mc.stationary_distributions[0]

						self.beta = np.inner(self.beta_hat, self.beta_stat)
						self.alpha_housing = np.inner(self.alpha_hat, self.alpha_stat)

						# Pension asset returns shock processes
						#lnrh_sd = parameters['sigma_d']*(parameters['h']**2)
						lnrh_sd = parameters['sigma_d']*parameters['h']
						lnrh_mc = tauchen(0, lnrh_sd, n=int(parameters['grid_size_DCR']))
						X_rh, P_rh = lnrh_mc.state_values, lnrh_mc.P[0]
						self.X_rh = np.exp(parameters['r_h'] + X_rh)
						self.P_rh = P_rh
						#lnrl_sd = parameters['sigma_d']*(parameters['l']**2)
						lnrl_sd = parameters['sigma_d']*parameters['l']

						lnrl_mc = tauchen(0, lnrl_sd,
															n=int(parameters['grid_size_DCR']))
						X_rl, P_rl = lnrl_mc.state_values, lnrl_mc.P[0]
						self.X_rl = np.exp(parameters['r_l'] + X_rl)
						self.P_rl = P_rl

						# Cartesian grid of realisatons from high and low risk asset
						self.X_r = cartesian([self.X_rl, self.X_rh])
						P_tmp = cartesian([self.P_rl, self.P_rh])

						# Joint probability array of high/low return realisations
						self.P_r = np.zeros(len(self.X_r))
						for i in range(len(self.P_r)):
								self.P_r[i] = P_tmp[i][0]*P_tmp[i][1]

						# housing return shocks
						self.Q_shocks_mc = tauchen(0, parameters['sigma_r_H'],
																			 n=int(parameters['grid_size_Q_s']))
						self.Q_shocks_r = self.Q_shocks_mc.state_values
						self.Q_shocks_P = self.Q_shocks_mc.P[0]

						# mortgage shocks
						self.lnmort_shocks = tauchen(0, parameters['kappa_m']*lnrl_sd,
															n=int(parameters['grid_size_DCR']))
						X_m, P_m = self.lnmort_shocks.state_values, self.lnmort_shocks.P[0]
						self.X_m = np.exp(parameters['r_m'] + X_m)

		class ModelCartGrids:

				"""
				Smaller cartesian grids for LS model 

				Attributes
				----------

				A_Q_R : 2D array
												cart. prod. house pr and liq assets for retirees A_RxQ
				X_Q_R_ind : 2D array
												cartesian grid of indices for A_RxQ
				A_R_H : 2D grid
												cart produc of H and A_R 
				H_Q : 2D array
												cart. product housing and house price grid HxQ
				X_H_R_ind: : 2D array
												cartesian grid of indices for H, Q and M, HxQxM
				X_R_cont_ind : 2D array
												cart product of indicies for retiree cont. states AxHxQxM
				Q_DC_shocks : 2D array
												Cart prod. of vals of house Q_shocks_r, X_rl, X_rh
				Q_DC_P : 2D array
												Transition prob. for 'Q_DC_shocks'
				EBA_P : 2D array
												Trans probs for the cart product W x alpha_hat x beta_hat
				EBA_shocks : 2D array
												Cart prod of W x alpha_hat x beta_hat shock vals

				"""

				def __init__(self, grid1d, stochgrids, param):

						stgrd = stochgrids
						self.A_Q_R = cartesian([grid1d.A_R, grid1d.Q])
						self.H_Q = cartesian([grid1d.H, grid1d.Q])
						self.HR_Q = cartesian([grid1d.H_R, grid1d.Q])
						self.A_R_H = cartesian([grid1d.A_R, grid1d.H])
						self.X_H_R_ind = cartesian([np.arange(param.grid_size_H),
																				np.arange(param.grid_size_Q)])
						self.X_R_cont_ind = cartesian([np.arange(param.grid_size_A),
																					 np.arange(param.grid_size_H),
																					 np.arange(param.grid_size_Q),
																					 np.arange(param.grid_size_M)])

						# Combine housing and pension return shocks
						# Q_DC_shocks[0] gives safe asset shock, Q_DC_shocks[1]
						# gives risky asset shock and Q_DC_shocks[2] gives housing shock
						self.Q_DC_shocks = cartesian([stgrd.X_rl, stgrd.X_rh, stgrd.Q_shocks_r,stgrd.X_m])

						P_tmp2 = cartesian([stgrd.P_rl, stgrd.P_rh,
																stgrd.Q_shocks_P,stgrd.X_m])
						self.Q_DC_P = np.zeros(len(self.Q_DC_shocks))

						for i in range(len(self.Q_DC_P)):
								self.Q_DC_P[i] = P_tmp2[i][0]*P_tmp2[i][1]*P_tmp2[i][2]

						self.EBA_P = np.zeros((int(param.grid_size_W),
																	 int(param.grid_size_alpha),
																	 int(param.grid_size_beta),
																	 int(param.grid_size_W
																			 * param.grid_size_alpha
																			 * param.grid_size_beta)))
						sizeEBA = int(param.grid_size_W
													* param.grid_size_alpha
													* param.grid_size_beta)
						self.EBA_P2 = self.EBA_P.reshape((sizeEBA, sizeEBA))

						for j in cartesian([np.arange(len(stochgrids.E)),
																np.arange(len(stochgrids.alpha_hat)),
																np.arange(len(stochgrids.beta_hat))]):

								EBA_P_temp = cartesian([stochgrids.P_E[j[0]],
																				stochgrids.P_alpha[j[1]],
																				stochgrids.P_beta[j[2]]])

								self.EBA_P[j[0], j[1], j[2], :]\
										= EBA_P_temp[:, 0]*EBA_P_temp[:, 1]*EBA_P_temp[:, 2]

						self.EBA_shocks = cartesian([stochgrids.E,
																				 stochgrids.alpha_hat,
																				 stochgrids.Q_shocks_r])

						# Make the mortgage rate shocks 
						# Standard deviation of ln_rm
						#sd_lnmort = np.sqrt(np.log(1 + (((beta_m*l*sigma_d)**2)/(r_m**2))))
						#mu_lnmort = np.log((r_m**2)/np.sqrt(r_m**2 + (beta_m*l*sigma_d)**2))
						
						# Normalise the standard deviation of the low DC return so it is 1 
						#dc_low_errors_sd_1 = self.Q_DC_shocks[:,0] - r_l -1
						#dc_low_errors_sd_1 = dc_low_errors_sd_1/np.sqrt((np.inner(dc_low_errors_sd_1**2, self.Q_DC_P)))

						# Construct the log-normally distribution r_m_prime shock vector 
						self.r_m_prime = self.Q_DC_shocks[:,3]
						#print(self.r_m_prime)

		class BigAssGrids:
				def __init__(self, grid1d, stochgrids, param):
						""" Large cartesian grids for the LS model

						X_all_hat_vals : 10D array
														Cartesian product of all worker state vals (not V)
														 DB/DC, E, Alpha, Beta, Pi, A, A_DC, H, Q,M
						X_all_hat_ind : 10D array
														 Cartesian product of exog. discrete and cont 
														 worker states DB/DC, E, Alpha, Beta, A, A_DC, H, Q, M
						X_all_C_vals : 9D array
														Cart product of of all worker state vals (not V, M)
						X_all_C_ind : 9D array
														Cart product of of all worker state indices (not V, M)
						X_all_B_vals : 8D array
														Cart product of of all worker state vals (not A,V, M)
						X_all_B_ind : 8D array
														Cart product of of all worker state vals (not A,V, M)

						"""
						# Construct large grids i.e. state-spaces
						self.stochgrids = stochgrids
						self.grid1d = grid1d
						self.param = param

				def X_all_hat_ind_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						X_all_hat_ind \
								= cartesian([grid1d.DB, np.arange(len(stochgrids.E)),
														 np.arange(len(stochgrids.alpha_hat)),
														 np.arange(len(stochgrids.beta_hat)),
														 np.arange(len(grid1d.Pi)),
														 np.arange(param.grid_size_A),
														 np.arange(param.grid_size_DC),
														 np.arange(param.grid_size_H),
														 np.arange(param.grid_size_Q),
														 np.arange(param.grid_size_M)])

						return X_all_hat_ind



				def X_all_C_ind_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						X_all_C_ind\
								= cartesian([grid1d.DB, np.arange(len(stochgrids.E)),
														 np.arange(len(stochgrids.alpha_hat)),
														 np.arange(len(stochgrids.beta_hat)),
														 np.arange(len(grid1d.Pi)),
														 np.arange(param.grid_size_A),
														 np.arange(param.grid_size_DC),
														 np.arange(param.grid_size_H),
														 np.arange(param.grid_size_Q)])
						return X_all_C_ind 

				def X_all_C_vals_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						X_all_C_vals\
								= cartesian([np.array([np.float64(grid1d.DB)]),
														 stochgrids.E,
														 stochgrids.alpha_hat,
														 stochgrids.beta_hat,
														 grid1d.Pi, grid1d.A,
														 grid1d.A_DC, grid1d.H,
														 grid1d.Q])
						return X_all_C_vals

				def X_all_B_ind_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						X_all_B_ind \
								= cartesian([grid1d.DB, np.arange(len(stochgrids.E)),
														 np.arange(len(stochgrids.alpha_hat)),
														 np.arange(len(stochgrids.beta_hat)),
														 np.arange(len(grid1d.Pi)),
														 np.arange(param.grid_size_DC),
														 np.arange(param.grid_size_HS),
														 np.arange(param.grid_size_Q)])
						return X_all_B_ind

				def X_all_B_vals_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						return cartesian([np.array([np.float64(grid1d.DB)]),
															stochgrids.E,
															stochgrids.alpha_hat,
															stochgrids.beta_hat,
															grid1d.Pi,
															grid1d.A_DC, grid1d.H_R,
															grid1d.Q])

				def X_V_func_DP_vals_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						return cartesian([grid1d.V,
															grid1d.A,
															grid1d.A_DC,
															grid1d.H,
															grid1d.M])

				def X_V_func_CR_vals_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						return cartesian([np.array([np.float64(grid1d.DB)]),
															stochgrids.E,
															stochgrids.alpha_hat,
															stochgrids.beta_hat,
															grid1d.Pi,
															grid1d.Q])

				def X_all_ind_f(self, ret_len=False):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						if ret_len == False:
								return cartesian([grid1d.DB,
												np.arange(len(stochgrids.E)),
												np.arange(len(stochgrids.alpha_hat)),
												np.arange(len(stochgrids.beta_hat)),
												np.arange(len(grid1d.V)),
												np.arange(len(grid1d.Pi)),
												np.arange(param.grid_size_A),
												np.arange(param.grid_size_DC),
												np.arange(param.grid_size_H),
												np.arange(param.grid_size_Q),
												np.arange(param.grid_size_M)])
						else:
								return int(len(grid1d.DB)*\
											len(stochgrids.E)*\
											len(stochgrids.alpha_hat)*\
											len(stochgrids.beta_hat)*\
											len(grid1d.V)*\
											len(grid1d.Pi)*\
											param.grid_size_A*\
											param.grid_size_DC*\
											param.grid_size_H*\
											param.grid_size_Q*\
											param.grid_size_M)


				def X_adj_func_ind_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						return cartesian([grid1d.DB,
															np.arange(len(stochgrids.E)),
															np.arange(len(stochgrids.alpha_hat)),
															np.arange(len(stochgrids.beta_hat)),
															np.arange(len(grid1d.Pi)),
															np.arange(param.grid_size_Q),
															np.arange(param.grid_size_M)])

				def X_nadj_func_ind_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						return cartesian([grid1d.DB,
															np.arange(len(stochgrids.E)),
															np.arange(len(stochgrids.alpha_hat)),
															np.arange(len(stochgrids.beta_hat)),
															np.arange(len(grid1d.Pi)),
															np.arange(param.grid_size_H),
															np.arange(param.grid_size_Q),
															np.arange(param.grid_size_M)])

				def X_W_bar_hdjex_ind_f(self):
						stochgrids = self.stochgrids
						grid1d = self.grid1d
						param = self.param
						return cartesian([grid1d.DB, np.arange(len(stochgrids.E)),
															np.arange(len(stochgrids.alpha_hat)),
															np.arange(len(stochgrids.beta_hat)),
															np.arange(len(grid1d.Pi)),
															np.arange(param.grid_size_DC),
															np.arange(param.grid_size_H),
															np.arange(param.grid_size_Q)])


class LifeCycleParams:
		""" Parameter class for the LifeCycleModel
						Instance of LifeCycleParams contains 
						parameter list, a paramterized LifeCycle model 
						for DB and a  LifeCycle model for DC

						Instance of Parameter class identified by
						unique parameter ID, param_id

						Todo
						----
						Should LifeCycle model contain paramter ID?

		"""

		def __init__(self,
								 mod_name,
								 param_dict,
								 scr_path, 
								 random_draw=False,
								 random_bounds=None,  # parameter bounds for randomly generated params
								 param_random_means=None,  # mean of random param distribution
								 param_random_cov=None,
								 uniform = False):  # cov of random param distribution :
				self.param_id = ''.join(random.choices(
						string.ascii_uppercase + string.digits, k=6))+'_'+time.strftime("%Y%m%d-%H%M%S") + '_'+mod_name
				self.mod_name = mod_name

				if random_draw == False:
						param_deterministic = param_dict['parameters']
						parameters_draws = rand_p_generator(param_deterministic,
																		random_bounds,
																		deterministic=1,
																		initial=uniform,
																		param_random_means=param_random_means,
																		param_random_cov=param_random_cov)
						self.og_DB = LifeCycleModel(param_dict, np.array(
								[0]), self.param_id, mod_name=mod_name)
						self.og_DC = LifeCycleModel(param_dict, np.array(
								[1]), self.param_id, mod_name=mod_name)
						self.parameters = parameters_draws
						self.parameters['param_id'] = self.param_id
						param_dict_new = copy.copy(self.parameters)

				if random_draw == True:
						param_deterministic = param_dict['parameters']

						parameters_draws = rand_p_generator(param_deterministic,
															random_bounds,
															deterministic = 0,
															initial = uniform,
															param_random_means = param_random_means,
															param_random_cov = param_random_cov)
						param_dict_new = copy.copy(param_dict)
						param_dict_new['parameters'] = parameters_draws

						self.og_DB = LifeCycleModel(param_dict_new, np.array(
								[0]), self.param_id, mod_name=mod_name)
						self.og_DC = LifeCycleModel(param_dict_new, np.array(
								[1]), self.param_id, mod_name=mod_name)
						self.parameters = parameters_draws
						self.parameters['param_id'] = self.param_id
						param_dict_new = copy.copy(self.parameters)
						
						
				#Path(scr_path + "/{}/{}_acc_0/".format(mod_name,self.param_id)).mkdir(parents=True, exist_ok=True)
				#pickle.dump(param_dict_new, open(scr_path + "/{}/{}_acc_0/params.smms".format(mod_name,self.param_id), "wb"))



def eval_rent_share(Q_shocks_r, r, r_H, r_l, beta_m):
		"""Calcuates rental to house price share

		Todo
		----

		Place this function in helper_funcs
		"""
		@njit
		def gen_npv_price():
				ph = np.ones((int(1e3)))

				for t in range(len(ph)):
						draws_Q = Q_shocks_r[np.random.randint(2)]
						ph[t+1] = ph[t]*(1 + r_H + draws_Q)/(1+r)

				return np.sum(ph)

		@njit(parallel=True, nogil=True)
		def gen_phi_r():
				npv_list = np.zeros(int(1e6))
				for j in prange(int(1e6)):
						npv_list[j] = gen_npv_price()

				return 1/np.mean(npv_list)

		#phi_r          = gen_phi_r()
		phi_r = .1
		return phi_r


if __name__ == "__main__":

		# Generate  instance of LifeCycleParams class with
		# an instance of DB LifeCycle model and DC LS model
		import yaml
		import csv

		# Folder contains settings (grid sizes and non-random params)
		# and random param bounds
		settings= 'settings/'
		# Name of model
		model_name = 'test'
		# Path for scratch folder (will contain latest estimated means)
		scr_path = "/scratch/pv33/ls_model_temp/"

		with open("{}settings.yml".format(settings), "r") as stream:
				eggbasket_config = yaml.safe_load(stream)

		param_random_bounds = {}
		with open('{}random_param_bounds.csv'.format(settings), newline='') as pscfile:
				reader_ran = csv.DictReader(pscfile)
				for row in reader_ran:
						param_random_bounds[row['parameter']] = np.float64([row['LB'],
																																row['UB']])

		sampmom = [0,1]
		model_name = 'baseline_male'
		#params = eggbasket_config['male']
		top_id = 'UDMK2H_20210924-175536_baseline_male'
		#params = pickle.load(open("/scratch/pv33/ls_model_temp2/baseline_male/{}_acc_0/params.smms".format(top_id),"rb"))
		#param_dict = eggbasket_config['male']
		#param_dict['parameters'] = params
		#param_dict['parameters']['nu_r_0']  = 0
		sampmom = pickle.load(open("/scratch/pv33/ls_model_temp2/baseline_male_v8/latest_sampmom.smms".format(model_name),"rb"))
		LS_models = LifeCycleParams('test',
																eggbasket_config['male'],
																scr_path,
																random_draw=False,
																random_bounds=param_random_bounds,
																param_random_means=sampmom[0],
																param_random_cov=sampmom[1],
																uniform = False)
		
