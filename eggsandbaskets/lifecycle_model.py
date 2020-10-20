
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

import numpy as np
from numba import njit, prange, guvectorize, jit
import time

import dill as pickle 
from sklearn.utils.extmath import cartesian
from quantecon import tauchen
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear


import warnings
warnings.filterwarnings('ignore')
from util.ls_model_functions import lsmodel_function_factory


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
               Account types for model 


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
                 mod_name = 'test', 
                 ):
        self.parameters = self.ModelParameters(config)
        self.functions = self.ModelFunctions(config)
        self.grid1d = self.Model1DGrids(acc_ind, config, self.parameters)
        self.interp_grid = self.ModelInterpGrids(self.parameters)
        self.st_grid = self.ModelStochProcGrids(config)
        self.cart_grids = self.ModelCartGrids(self.grid1d,
                                              self.st_grid,
                                              self.parameters)
        self.BigAssGrids = self.BigAssGrids(self.grid1d, self.st_grid, self.parameters)

        self.acc_ind = acc_ind
        self.mod_name = mod_name

        #self.parameters.phi_r = eval_rent_share(self.st_grid.Q_shocks_r,
       #                                        self.parameters.r,
       #                                         self.parameters.r_H,
       #                                         self.parameters.r_l,
       #                                         self.parameters.beta_m)

        self.ID = time.strftime("%Y%m%d-%H%M%S") +'_'+mod_name+'_' + format(acc_ind[0])

        

    class ModelParameters:

        """ All model parameters and grid sizes
        """

        def __init__(self, config):
            parameters = config['parameters']
            lambdas = np.array(config['lambdas'])
            normalisation = np.array(config['normalisation'])

            self.__dict__.update(config['parameters'])

            # Change type of
            # survival probabilties terminal age and retirement age to int
            self.s = np.array(config['survival'])
            self.T = int(self.T)

            # overide pre-set welath and mortgage max
            self.M_max = self.H_max*self.Q_max*(1-self.phi_c)
            self.A_max_WW = self.H_max*self.Q_max + self.A_max_W
            self.A_max_WE = self.H_max*self.Q_max + self.A_max_R

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
                self.u_vec, self.uc_vec\
                = lsmodel_function_factory(parameters,
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
                         (param.M_min, param.M_max, param.grid_size_M),
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
            self.X_cont_WAM\
                = UCGrid((param.A_min, param.A_max_W, param.grid_size_A),
                         (param.M_min, param.M_max, param.grid_size_M))
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
            self.beta_mc = tauchen(parameters['rho_beta'],
                                   parameters['sigma_beta'],
                                   n=int(parameters['grid_size_beta']))
            self.beta_hat = self.beta_mc.state_values
            self.P_beta = self.beta_mc.P
            self.beta_stat = self.beta_mc.stationary_distributions[0]
            self.alpha_mc = tauchen(parameters['rho_alpha'],
                                    parameters['sigma_alpha'],
                                    n=int(parameters['grid_size_alpha']))
            self.alpha_hat, self.P_alpha = self.alpha_mc.state_values, \
                self.alpha_mc.P
            self.alpha_stat = self.alpha_mc.stationary_distributions[0]


            self.beta = np.inner(np.exp(self.beta_hat
                                        + np.log(parameters['beta_bar'])),
                                 self.beta_stat)
            self.alpha_housing = np.inner(np.exp(self.alpha_hat
                                                 + np.log(parameters['alpha_bar'])),
                                          self.alpha_stat)

            # Pension asset returns shock processes
            #lnrh_sd = parameters['sigma_d']*(parameters['h']**2)
            lnrh_sd = parameters['sigma_d']*parameters['h']
            lnrh_mc = tauchen(0, lnrh_sd,
                              n=int(parameters['grid_size_DCR']))
            X_rh, P_rh = lnrh_mc.state_values, lnrh_mc.P[0]
            self.X_rh = np.exp(np.log(parameters['r_h']) + X_rh)
            self.P_rh = P_rh
            #lnrl_sd = parameters['sigma_d']*(parameters['l']**2)
            lnrl_sd = parameters['sigma_d']*parameters['l']

            lnrl_mc = tauchen(0, lnrl_sd,
                              n=int(parameters['grid_size_DCR']))
            X_rl, P_rl = lnrl_mc.state_values, lnrl_mc.P[0]
            self.X_rl = np.exp(np.log(parameters['r_l']) + X_rl)
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
                                        np.arange(param.grid_size_Q),
                                        np.arange(param.grid_size_M)])
            self.X_R_cont_ind = cartesian([np.arange(param.grid_size_A),
                                           np.arange(param.grid_size_H),
                                           np.arange(param.grid_size_Q),
                                           np.arange(param.grid_size_M)])

            # combine housing and pension return shocks
            # Q_DC_shocks[0] gives safe asset shock, Q_DC_shocks[1]
            # gives risky asset shock and Q_DC_shocks[2] gives housing shock
            self.Q_DC_shocks = cartesian([stgrd.X_rl, stgrd.X_rh,
                                          stgrd.Q_shocks_r])
            P_tmp2 = cartesian([stgrd.P_rl, stgrd.P_rh,
                                stgrd.Q_shocks_P])
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

    class BigAssGrids:
        def __init__(self,grid1d, stochgrids, param):
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
            self.param =param




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

        def  X_all_B_vals_f(self):
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
        def X_all_ind_f(self, ret_len = False):
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
            return len(cartesian([grid1d.DB,
                  np.arange(len(stochgrids.E)),
                  np.arange(len(stochgrids.alpha_hat)),
             np.arange(len(stochgrids.beta_hat)),
             np.arange(len(grid1d.V)),
             np.arange(len(grid1d.Pi)),
             np.arange(param.grid_size_A),
             np.arange(param.grid_size_DC),
             np.arange(param.grid_size_H),
             np.arange(param.grid_size_Q),
             np.arange(param.grid_size_M)]))

        
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
                       np.arange(param.grid_size_HS),
                       np.arange(param.grid_size_Q),
                       np.arange(param.grid_size_M)])

def plot_policy_W(policy, og, acc_ind):

    plots_folder = '/home/141/as3442/temp_plots/'
    NUM_COLORS = len(og.grid1d.M)
    colormap = cm.viridis
    M_vals = np.linspace(0, og.grid1d.M[-1], 15)
    normalize = mcolors.Normalize(vmin=np.min(M_vals), vmax=np.max(M_vals))


    for age in np.arange(int(og.parameters.tzero), int(og.parameters.R)):
      for pol_key, key in zip([0,1,2],('c_adj','H_adj','a_adj')):
        for l in range(len(og.grid1d.Q)):
            for k in range(len(og.grid1d.A_DC)):
                t = int(age - og.parameters.tzero)
                pols_adj = policy[int(3 + pol_key)][t][acc_ind, 1, 1, 1, 1, k, l, :, :]
                plt.close()
                NUM_COLORS = len(pols_adj)
                colormap = cm.viridis
                fig = plt.figure()
                ax = plt.axes()
                for i in range(len(pols_adj)):
                    ax.plot(og.grid1d.W_W, pols_adj[i], color=colormap(i//3*3.0/NUM_COLORS))
                    ax.set_xlabel('Total wealth (AUD 100000)')
                    ax.set_ylabel('Housing  (after adjusting)')
                    ax.set_title('DC balance {} and house price {}'.format(
                        round(og.grid1d.A_DC[k], 1), round(og.grid1d.Q[l], 1)))

                scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
                scalarmappaple.set_array(M_vals)
                cbar = plt.colorbar(scalarmappaple)
                cbar.set_label('Mort. liability')

                plt.savefig(
                    '{}/{}_W_{}_DC{}_P{}.png'.format(plots_folder, age, key, k, l))

                plt.close()



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

    import yaml
    import dill as pickle

    # housing model modules
    from util.grids_generate import generate_points
    from util.helper_funcs import *
    from solve_policies.worker_solver import generate_worker_pols
    from util.randparam import rand_p_generator
    from generate_timeseries.tseries_generator import gen_panel_ts, gen_moments, sortmoments, genprofiles_operator
    import copy

    from matplotlib.colors import DivergingNorm
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    # read settings
    with open("settings.yml", "r") as stream:
        eggbasket_config = yaml.safe_load(stream)

        import ray 
    ray.init(num_cpus = 2)

    @ray.remote
    def run_acc_in(acc_ind):
      # Create housing model
      if acc_ind == 0:
        time.sleep(45)
      og = LifeCycleModel(eggbasket_config['baseline_lite'],
                          np.array([acc_ind]))

      # Solve model
      policies = generate_worker_pols(og, load_retiree=1)

      #plot_policy_W(policies, og)

      return policies



    results_DB_id = run_acc_in.remote(0)
    results_DC_id = run_acc_in.remote(1)

    results_DB, results_DC = ray.get([results_DB_id,results_DC_id])

    pickle.dump(results_DB, \
            open("/scratch/pv33/ls_model_temp/baseline_DBID.pols", "wb"))


    pickle.dump(results_DC, \
           open("/scratch/pv33/ls_model_temp/baseline_DCID.pols", "wb"))
    

    og = LifeCycleModel(eggbasket_config['baseline_lite'],
                          np.array([acc_ind]))


    #joined_pols = pickle.load(open("/scratch/pv33/ls_model_temp/baseline_lite.pols", "rb"))


    TSN = 100
    U = np.random.rand(6,100,TSN,100) 

    TSALL_10_df, TSALL_14_df = gen_panel_ts(og, joined_pols,U, TSN)
    
    moments_male        = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_male') 

    moments_female      = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_female')


    moments_sim_sorted    = sortmoments(moments_male,\
                                         moments_female)

    