"""
Module contains function to generate 
state-space points at which 
functions are interpolated 
for a housing model class"""

# Import packages

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from quantecon import tauchen

from itertools import product
from numba import njit, prange, guvectorize, jit
import numba

from sklearn.utils.extmath import cartesian 

from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interpolation import interp

from pathos.multiprocessing import Pool as ProcessPool
from quantecon.optimize.root_finding import brentq

from interpolation.splines import extrap_options as xto


from collections import defaultdict

from numpy.core.umath_tests import inner1d

import time


def generate_points(og, scratch = True):

    amort_rate          = og.functions.amort_rate

    V,Pi, M,H, Q, A     = og.grid1d.V, og.grid1d.Pi, og.grid1d.M, og.grid1d.H, og.grid1d.Q, og.grid1d.A
    v_S, v_E            = og.parameters.v_S, og.parameters.v_E
    A_DC                = og.grid1d.A_DC
    A_min               = og.parameters.A_min

    tau_housing         = og.parameters.tau_housing
    delta_housing       = og.parameters.delta_housing

    R                   = og.parameters.R
    tzero               = og.parameters.tzero
    yvec                = og.functions.yvec
    E                   = og.st_grid.E
    r                   = og.parameters.r


    r_H                 = og.parameters.r_H
    r_l                 = og.parameters.r_l
    beta_m              = og.parameters.beta_m
    kappa_m             = og.parameters.kappa_m

    alpha_bar           = og.parameters.alpha_bar
    beta_bar            = og.parameters.beta_bar
    alpha_hat           = og.st_grid.alpha_hat
    beta_hat            = og.st_grid.beta_hat

    EBA_P               = og.cart_grids.EBA_P

    X_all_ind           = og.BigAssGrids.X_all_ind_f()
    Q_DC_shocks         = og.cart_grids.Q_DC_shocks
    X_all_hat_ind       = og.BigAssGrids.X_all_hat_ind_f()


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

    @njit 
    def gen_x_prime_vals(i):
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
        """
        r_share         = Pi[X_all_hat_ind[i][4]]
        ADC_in          = A_DC[X_all_hat_ind[i][6]]
        H_in            = H[X_all_hat_ind[i][7]]
        q_in            = Q[X_all_hat_ind[i][8]]
        m_in            = M[X_all_hat_ind[i][9]]

        DB_ind              = 0
        E_ind               = X_all_hat_ind[i][1]
        alpha_ind           = X_all_hat_ind[i][2]
        beta_ind            = X_all_hat_ind[i][3]
        a_ind               = int(X_all_hat_ind[i][5])
        h_ind               = int(X_all_hat_ind[i][7])
        q_ind               = X_all_hat_ind[i][8]

        q_t_arr             = np.full(len(Q_DC_shocks[:,2]),q_in)
        r_H_arr             = np.full(len(Q_DC_shocks[:,2]),r_H)
        Q_prime             = 1+r_H_arr + Q_DC_shocks[:,2]

        risky_share_arr     = np.full(len(Q_DC_shocks[:,2]),r_share)

        A_DC_returns        = 1+(1-risky_share_arr)*Q_DC_shocks[:,0] +\
                                risky_share_arr*Q_DC_shocks[:,1]

        A_DC_prime          = A_DC_returns*\
                                np.full(len(Q_DC_shocks[:,2]),ADC_in)

        r_m_prime           = beta_m*r_l*(Q_DC_shocks[:,0]/r_l)**kappa_m 
        M_prime             = (1+r_m_prime)*m_in 

        return np.column_stack((A_DC_prime,Q_prime,M_prime,r_m_prime))

    @njit
    def gen_x_prime_array():

        X_prime_vals            = np.empty((len(X_all_hat_ind),\
                                    len(Q_DC_shocks[:,2]), 4))

        for i in prange(len(X_all_hat_ind)):
            X_prime_vals[i,:]       = gen_x_prime_vals(i)

        return X_prime_vals

    @njit
    def gen_alph_beta(i):
        alpha_hs = np.exp(alpha_hat[X_all_ind[i][2]]\
                            + np.log(alpha_bar))
        beta = np.exp(beta_hat[X_all_ind[i][3]]\
                            + np.log(beta_bar))

        return np.array([alpha_hs, beta])

    @njit
    def gen_alpha_beta_array():
        X_all_ind_W_vals = np.empty((len(X_all_ind), 2))

        for i in prange(len(X_all_ind)):

            """vals in X_all_ind_W_vals are:

            0 - alpha
            1 - beta"""

            X_all_ind_W_vals[i,:]   =  gen_alph_beta(i)

        return X_all_ind_W_vals



    @njit
    def gen_wage_vec():
        wage_vector                 = np.empty((int(R-tzero), len(X_all_ind[:,1])))

        for j in prange(int(R-tzero)):
            wage_vector[j,:]    = yvec(int(j+ tzero), E[X_all_ind[:,1]] )

        return wage_vector


    @njit
    def gen_RHS_points():

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
        wage_vector                    = gen_wage_vec()

        points_noadj_vec, points_adj_vec, points_rent_vec\
                    = np.empty((int(R-tzero), len(X_all_ind[:,1]),2)),\
                        np.empty((int(R-tzero), len(X_all_ind[:,1]),2)),\
                        np.empty((int(R-tzero), len(X_all_ind[:,1]),2))

        A_prime     = np.empty((int(R-tzero), len(X_all_ind[:,1])))


        acc_ind             = np.full(len(X_all_ind[:,4]), int(0))
        v_Sv                = np.full(len(X_all_ind[:,4]), v_S)
        v_Ev                = np.full(len(X_all_ind[:,4]), v_E)
        v                   = V[X_all_ind[:,4]]

        # 1- total contribute rate as % of wage for all states 
        contrat             = np.ones(len(X_all_ind[:,4])) \
                                - v -v_Sv - v_Ev

        pi                  = Pi[X_all_ind[:,5]]
        m                   = M[X_all_ind[:,10]]
        h                   = H[X_all_ind[:,8]]*(1-delta_housing)
        q                   = Q[X_all_ind[:,9]]

        tau_housing_vec     = np.full(len(X_all_ind[:,4]), tau_housing)

        for j in prange(int(R-tzero)):

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
            
            amort_rate_vec      = np.full(len(X_all_ind[:,4]),\
                                        amort_rate(int(j+ tzero-2)) )

            wage                = wage_vector[j]

            # total liquid wealth (cash in hand) for non-adjuster
            # and adjuster 
            points_noadj_vec[j,:,0]         = A[X_all_ind[:,6]]*(1+r)\
                                                + contrat*wage\
                                                - amort_rate_vec*m
            points_noadj_vec[j,:,0][points_noadj_vec[j,:,0]<=0] = A_min

            points_adj_vec[j,:,0]           = points_noadj_vec[j,:,0]\
                                                 + q*h
            points_adj_vec[j,:,0][points_adj_vec[j,:,0]<=0] = A_min
            # next period DC assets (before returns)
            # (recall domain of policy functions from def of eval_policy_W)
            points_noadj_vec[j,:,1]             = A_DC[X_all_ind[:,7]]\
                                                    + v*wage\
                                                    + (v_S +v_E)*wage\
                                                        *X_all_ind[:,0]
            points_adj_vec[j,:,1]               = A_DC[X_all_ind[:,7]]\
                                                    + v*wage\
                                                    + (v_S +v_E)*wage\
                                                        *X_all_ind[:,0]
            # renter cash at hand (after mortgage deduction)
            points_rent_vec[j,:,0]              = points_noadj_vec[j,:,0]\
                                                     + q*\
                                                        (h-h*tau_housing_vec)\
                                                         - m                # should mortgage go here?
            points_rent_vec[j,:,0][points_rent_vec[j,:,0]<=0] = A_min

            points_rent_vec[j,:,1]              = A_DC[X_all_ind[:,7]]\
                                                    + v*wage\
                                                    + (v_Sv +v_Ev)\
                                                        *wage*X_all_ind[:,0]

            A_prime[j]                          =  points_adj_vec[j,:,0]\
                                                     - m
            A_prime[j][A_prime[j]<=0]           = 1e-200 

        # reshape the adjustment points to wide
        # recall the points adj_vec are ordered accordint to
        # X_all_ind  
        points_adj_vec1         = points_adj_vec.reshape((int(R-tzero),\
                                                        1,grid_size_W,\
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
        #  Age (0), DB(1),E(2), Alpha(3), Beta(4), Pi(6), Q(10),\
        #  M(11), V(5), A(7), A_DC(8), H(9), 2D points(12)

        points_adj_vec2 = np.copy(points_adj_vec1).transpose((0,1,2,3,4,\
                                                6,10,11,5,7,8,9,12))

        # reshape to:
        # Age, |DBx Ex Alpha x BetaxPi xQxM| x|VxAxA_DCxH|x2
        # recall each i \in |DBx Ex Alpha x BetaxPi xQxM|
        # adjuster function will be reshaped to a
        # function on Wealth x A_DC 
        points_adj_vec3 = np.copy(points_adj_vec2).reshape((int(R-tzero),\
                                                    int(1*grid_size_W*grid_size_alpha*
                                                    grid_size_beta*len(Pi)*\
                                                    grid_size_Q*grid_size_M),\
                                                    int(grid_size_H*len(V)*\
                                                    grid_size_A*\
                                                    grid_size_DC),2))
        # reshape the renter points  
        points_rent_vec1 = points_rent_vec.reshape((int(R-tzero),\
                                                    1,grid_size_W,\
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
        #  Age (0),DB(1), E(2), Alpha(3), Beta(4), Pi(6), Q(10),\
        #  V(5), A(7), A_DC(8), H(9), M(11), points(2)
        points_rent_vec2 = np.copy(points_rent_vec1)\
                                .transpose((0,1,2,3,4,6,10,5,7,8,9,11,12))

        points_rent_vec3 = np.copy(points_rent_vec2).reshape((int(R-tzero),\
                                                    int(1*grid_size_W*grid_size_alpha*
                                                    grid_size_beta*len(Pi)*\
                                                    grid_size_Q),\
                                                    int(grid_size_H*len(V)*\
                                                    grid_size_A*grid_size_M*\
                                                    grid_size_DC),2))

        # reshape the no adjustment points  
        points_noadj_vec1 = np.copy(points_noadj_vec).reshape((int(R-tzero),\
                                                    1,grid_size_W,\
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
        #  Age (0),DB(1), E(2), Alpha(3), Beta(4), Pi(6), H(9), Q(10),M(11)\
        #  V(5), A(7), A_DC(8), points(12)
        points_noadj_vec2 = np.copy(points_noadj_vec1).transpose((0,1,2,3,4,6,9,10,11,5,7,8,12))
        points_noadj_vec3 = np.copy(points_noadj_vec2).reshape((int(R-tzero),\
                                            int(1*grid_size_W*grid_size_alpha*
                                            grid_size_beta*len(Pi)*\
                                            grid_size_H*grid_size_Q*grid_size_M),\
                                            int(len(V)*
                                            grid_size_A*
                                            grid_size_DC),2))

        return points_noadj_vec3, points_adj_vec3,\
                 points_rent_vec3,A_prime


    X_prime_vals        = gen_x_prime_array()

    #EBA_P_mat              = gen_EBA_P_mat()

    X_all_ind_W_vals    = gen_alpha_beta_array()


    points_noadj_vec, points_adj_vec, points_rent_vec, A_prime = gen_RHS_points()


    if scratch ==True:

        for Age in np.arange(int(tzero), int(R))[::-1]:
            np.savez_compressed("/scratch/pv33/ls_model_temp/grigrid_modname_{}_age_{}".format(og.mod_name, Age),\
                                    points_noadj_vec  = points_noadj_vec[int(Age-tzero)],\
                                    points_adj_vec = points_adj_vec[int(Age-tzero)],\
                                    points_rent_vec = points_rent_vec[int(Age-tzero)],\
                                    A_prime = A_prime[int(Age-tzero)])


            np.savez_compressed("/scratch/pv33/ls_model_temp/grigrid_modname_{}_genfiles".format(og.mod_name, Age),\
                        X_all_ind_W_vals  = X_all_ind_W_vals,\
                        X_prime_vals = X_prime_vals)

        return 0

    else:
        return  X_all_ind_W_vals, X_prime_vals, points_noadj_vec, points_adj_vec, points_rent_vec, A_prime