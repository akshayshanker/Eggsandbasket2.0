
"""
Module generates operator to solve worker policy functions

Functions: worker_solver_factory
			Generates the operators require to solve an instance \
			of Housing Model

Example use:

# solve model
gen_R_pol       = retiree_func_factory(lifecycle_model)

solve_LC_model  = worker_solver_factory(lifecycle_model,
							gen_R_pol)

policy          = solve_LC_model()

"""

# Import packages

import dill as pickle
from sklearn.utils.extmath import cartesian
from interpolation import interp
from interpolation.splines import extrap_options as xto
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from itertools import product
from quantecon.optimize.root_finding import brentq
from quantecon import tauchen
import numba
from numba import njit, prange, guvectorize, jit
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("..")
from solve_policies.retiree_solver import retiree_func_factory
from eggsandbaskets.util.helper_funcs import gen_policyout_arrays, d0, interp_as,\
    gen_reshape_funcs, einsum_row
from eggsandbaskets.util.grids_generate import generate_points


def worker_solver_factory(og, gen_R_pol):
    """ Generates solver of worker policies

    Parameters
    ----------
    og : LifeCycleModel
            Instance of LifeCycleModel
    gen_R_pol: Function
            function that generates retiring (T_R) age policy and RHS Euler

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
    u = og.functions.u
    uc = og.functions.uc
    uh = og.functions.uh
    uc_inv = og.functions.uc_inv
    uh_inv = og.functions.uh_inv

    u_vec, uc_vec = og.functions.u_vec, og.functions.uc_vec
    amort_rate = og.functions.amort_rate
    b, b_prime = og.functions.b, og.functions.b_prime
    y, yvec = og.functions.y, og.functions.yvec
    adj_p, adj_v, adj_pi = og.functions.adj_p, og.functions.adj_v,\
        og.functions.adj_pi

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

    EBA_P, EBA_P2 = og.cart_grids.EBA_P, og.cart_grids.EBA_P2

    E, P_E, P_stat = og.st_grid.E, og.st_grid.P_E, og.st_grid.P_stat

    # smaller, "fundamental" grids
    A_DC = og.grid1d.A_DC
    H =  og.grid1d.H
    A = og.grid1d.A
    M = og.grid1d.M
    Q =  og.grid1d.Q
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
    X_W_contgp = og.interp_grid.X_W_contgp

    # larger grids
    X_V_func_DP_vals = og.big_grids.X_V_func_DP_vals
    X_V_func_CR_vals = og.big_grids.X_V_func_CR_vals
    X_W_bar_hdjex_vals = og.big_grids.X_W_bar_hdjex_vals
    X_W_bar_hdjex_ind = og.big_grids.X_W_bar_hdjex_ind
    X_all_hat_ind = og.big_grids.X_all_hat_ind
    X_all_C_ind = og.big_grids.X_all_C_ind
    X_all_B_ind = og.big_grids.X_all_B_ind
    X_all_hat_vals = og.big_grids.X_all_hat_vals
    X_all_C_vals = og.big_grids.X_all_C_vals
    X_all_B_vals = og.big_grids.X_all_B_vals
    X_all_ind = og.big_grids.X_all_ind
    X_adj_func_ind = og.big_grids.X_adj_func_ind
    X_nadj_func_ind = og.big_grids.X_nadj_func_ind

    # empty arrays to store policy function outputs
    all_state_shape, all_state_shape_hat, v_func_shape,\
        all_state_A_last_shape, policy_shape_nadj, policy_shape_adj,\
        policy_Aprime_noadj, policy_C_noadj, policy_etas_noadj,\
        policy_Aprime_adj, policy_C_adj, policy_H_adj,\
        policy_Xi_cov, policy_Xi_copi, policy_VF,\
        policy_zeta, policy_H_rent\
        = gen_policyout_arrays(og)

    # generate pre-filled interpolation points
    X_all_ind_W_vals, X_prime_vals, points_noadj_vec,\
        points_adj_vec, points_rent_vec, A_prime \
        = generate_points(og)

    # generate all the re-shaping functions
    reshape_out_pi, reshape_out_V, reshape_X_bar,\
        reshape_make_Apfunc_last, reshape_make_h_last,\
        reshape_vfunc_points, reshape_nadj_RHS, reshape_rent_RHS,\
        reshape_adj_RHS, reshape_RHS_Vfunc_rev, reshape_RHS_UFB,\
        reshape_vfunc_points = gen_reshape_funcs(og)

    # Define functions that return errors of first order conditions

    @njit
    def mort_FOC(m, UC_prime_func1D, UC_prime_M_func1D):
        """ Error of first order cond. of mortgage interior solution
                (equation x in paper)

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

        error = np.interp(m,M,UC_prime_func1D)\
            - np.interp(m,M, UC_prime_M_func1D)
        return error

    @njit
    def liq_rent_FOC_W(cons, h, q, alpha_housing, UC_prime_func1D):
        """ Solves frst order condition of liq. asset choice of renters
                (equation x in paper)

        Parameters
        ----------
        cons : float64
                consumption at period t
        h : float64
                housing services consumed at t
        alpha_housing : float64
                housing share parameters
        UC_prime_func1D : 1D array
                Interpolant of cond. expec. of marginal u of wrt to a_t+1

        Returns
        -------
        a_prime : float64
                Optimal interior a_t+1 implied by FOC

        Notes
        -----

        The marginal utility is an interpolant conditoned on end of
        time t information as "function" of a_t+1. See notes for
        `UC_cond_all' for states the marginal utility is conditioned
        on

        The RHS marginal utility  should assumes h_t taken into
        period t+1 is 0

        """

        LHS_val = uc(cons, h, alpha_housing)
        a_prime =  interp_as(UC_prime_func1D, A,np.array([LHS_val]))[0]

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
               UC_prime_HFC_func2D,
               UC_prime_M_func2D,
               t, ret_cons=False):
        """ Returns Euler error for housing first order condition
                (Equation x in paper) where liq assett FOC holds
                and c_t implicit

        Parameters
        ----------
        x_prime : float64
                a_t+1 next period liquid
        a_dc : float64
                before returns t+1 DC assets
        h : float64
                H_t
        q : float64
                P_t house price
        m : float64
                M_t mort liability (after r)
        beta : float64
                time t discount rate
        alpha_housing : float64
                time t housing share
        wage : float64
                wage at time t and state
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
        ret_cons :Bool
                if True, returns c_t associated with  `x_prime' and `m_prime'

        Returns
        -------
        Euler error : float64

        Notes
        -----

        The marginal utilities are interpolants conditoned on end of
        time t information as "function" of a_t+1 and m_t+1

        See notes for  `UC_cond_all' for states the marginal utility
        should be conditioned on

        Todo
        ----

        Verify if `ref_const_FOC' should be inside or outside the
        absolute value sign

        """

        max_loan = min(
            q * h * (1 - phi_c), phi_d * wage / amort_rate(t - 1))

        uc_prime_maxloan = beta * eval_linear(X_cont_WAM,
                                              UC_prime_func2D,
                                              np.array([x_prime, max_loan]),
                                              xto.LINEAR)

        m_prime_adj = interp_as(A, mort_func1D, np.array([x_prime]))[0]
        m_prime = max(0, min(m_prime_adj, max_loan))
        uc_prime_adj = max(1e-250,
                           beta * eval_linear(X_cont_WAM,
                                              UC_prime_func2D,
                                              np.array([x_prime, m_prime]),
                                              xto.LINEAR))
        c_t = max(C_min, uc_inv(uc_prime_adj, h, alpha_housing))

        # Indictor for whether actual constrained mortage is
        # constrained by collat. constraint
        ref_const_H = m_prime == q * h * (1 - phi_c)

        # interp values of RHS of housing Euler
        UC_prime_RHS = beta * eval_linear(X_cont_WAM, UC_prime_func2D,
                                          np.array([x_prime, m_prime]), xto.LINEAR)
        UC_prime_H_RHS = beta * eval_linear(X_cont_WAM, UC_prime_H_func2D,
                                            np.array([x_prime, m_prime]), xto.LINEAR)
        UC_prime_HFC_RHS = beta * eval_linear(X_cont_WAM, UC_prime_HFC_func2D,
                                              np.array([x_prime, m_prime]), xto.LINEAR)

        # Calculate the marginal value of relaxising the collateral
        # constraint (ref_const_FOC)
        # Recall this is marginal utility of consumption today
        # - marginal cost of borrowing today (RHS of Euler wrt to m_t+1)
        # We only care about this value if it is positive i.e.
        # for negative values, the collat constraint will never bind
        ref_const_FOC = ref_const_H\
            * q * (1 - phi_c)\
            * (max(uc(c_t, h, alpha_housing)
                   - uc_prime_maxloan, 0))

        RHS = uc(c_t, h, alpha_housing) * q * (1 + tau_housing)\
            - UC_prime_H_RHS

        if ret_cons:
            return c_t

        # Return equation x in paper

        if UC_prime_HFC_RHS - ref_const_FOC != 0:
            return np.abs((uh(c_t, h, alpha_housing) + ref_const_FOC - RHS))\
                - UC_prime_HFC_RHS
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
              UC_prime_HFC_func2D,
              UC_prime_M_func2D,
              t):
        """ Returns error of housing FOC
        with liquod assett FOC binding and c_t explicit

        Equation x in paper.

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
        uc_prime_fullpay = beta * eval_linear(X_cont_WAM,
                                              UC_prime_func2D,
                                              np.array([x_prime, 0]),
                                              xto.LINEAR)
        uc_prime_maxloan = beta * eval_linear(X_cont_WAM,
                                              UC_prime_func2D,
                                              np.array([x_prime, max_loan]),
                                              xto.LINEAR)
        ucmprime_m_fullpay = beta * eval_linear(X_cont_WAM,
                                                UC_prime_M_func2D,
                                                np.array([x_prime, 0]),
                                                xto.LINEAR)
        ucmprime_m_maxloan = beta * eval_linear(X_cont_WAM,
                                                UC_prime_M_func2D,
                                                np.array([x_prime,
                                                          max_loan]), xto.LINEAR)

        if uc_prime_maxloan > ucmprime_m_maxloan:
            m_prime = max_loan
        else:
            m_prime = 0

        # Calcuate RHS of Euler at m_prime

        UC_prime_M_RHS = beta * eval_linear(X_cont_WAM, UC_prime_M_func2D,
                                            np.array([x_prime, m_prime]),
                                            xto.LINEAR)
        UC_prime_RHS = beta * eval_linear(X_cont_WAM, UC_prime_func2D,
                                          np.array([x_prime, m_prime]),
                                          xto.LINEAR)
        UC_prime_H_RHS = beta * eval_linear(X_cont_WAM, UC_prime_H_func2D,
                                            np.array([x_prime, m_prime]),
                                            xto.LINEAR)
        UC_prime_HFC_RHS = beta * eval_linear(X_cont_WAM, UC_prime_HFC_func2D,
                                              np.array([x_prime, m_prime]),
                                              xto.LINEAR)

        # Calcuate indictor for collat constraint holding
        ref_const_H = m_prime == q * h * (1 - phi_c)

        # Marginal value of relaxing collat constraint
        ref_const_FOC = ref_const_H\
            * q * (1 - phi_c)\
            * (max(0,
                   uc(c, h, alpha_housing) - UC_prime_M_RHS))

        RHS = uc(c, h, alpha_housing) * q * (1 + tau_housing)\
                - UC_prime_H_RHS

        if UC_prime_HFC_RHS - ref_const_FOC != 0:
            return np.abs((uh(c, h, alpha_housing) - RHS+ref_const_FOC))\
                    - UC_prime_HFC_RHS
        else:
            return uh(c, h, alpha_housing) - RHS

    # Define functions that solve the policy functions

    @njit
    def eval_M_prime(UC_prime_func1D, UC_prime_M_func1D):
        """ Solve mortgage FOC for unconstrained mortgage solution"""

        UC_prime_m_max = UC_prime_func1D[-1]
        UC_prime_M_m_max = UC_prime_M_func1D[-1]
        UC_prime_m_min = UC_prime_func1D[0]
        UC_prime_M_m_min = UC_prime_M_func1D[0]

        if mort_FOC(0, UC_prime_func1D, UC_prime_M_func1D)\
                * mort_FOC(M[-1], UC_prime_func1D, UC_prime_M_func1D) < 0:
            return brentq(mort_FOC, 0, M[-1],
                          args=(UC_prime_func1D,
                                UC_prime_M_func1D), xtol = 1e-05)[0]
        elif UC_prime_m_max >= UC_prime_M_m_max:
            return M[-1]
        else:
            return 0

    @njit
    def eval_pol_mort_W(t, UC_prime, UC_prime_M):
        """
        Evaluates unconstrained  mortgage policy function
        as function of H_t,DC_t+1, A_t+1 and exogenous states at t"""

        mort_prime = np.zeros(len(X_all_C_vals))
        UC_prime_func_ex = UC_prime.reshape(all_state_shape)
        UC_prime_M_func_ex = UC_prime_M.reshape(all_state_shape)

        for i in prange(len(X_all_C_vals)):
            h, h_ind = X_all_C_vals[i][7], int(X_all_C_ind[i][7])
            a_prime, ap_ind = X_all_C_vals[i][5], int(X_all_C_ind[i][5])
            q, q_ind = X_all_C_vals[i][8], int(X_all_C_ind[i][8])

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
            mort_prime[i] = eval_M_prime(UC_prime_func1D,
                                         UC_prime_M_func1D)

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
        """
        Creates renter policy function as function of
        wealth_t, q_t, dc_t+1 and exogenous states at t using endog
        grid method

        """

        a_end_1 = np.zeros(len(X_all_B_ind))  # Endog grid for a_t
        UC_prime_func_ex = UC_prime.reshape(all_state_shape)

        for i in prange(len(X_all_B_ind)):

            h, h_ind = X_all_B_vals[i][6], int(X_all_B_ind[i][6])
            q, q_ind = X_all_B_vals[i][7], int(X_all_B_ind[i][7])

            E_ind = int(X_all_B_ind[i][1])
            alpha_ind = int(X_all_B_ind[i][2])
            beta_ind = int(X_all_B_ind[i][3])
            Pi_ind = int(X_all_B_ind[i][4])
            DB_ind = 0
            adc_ind = int(X_all_B_ind[i][5])

            beta = np.exp(X_all_B_vals[i][3]
                          + np.log(beta_bar))
            alpha_housing = np.exp(X_all_B_vals[i][2]
                                   + np.log(alpha_bar))

            # get UC_prime as interpolant on a_t+1
            # next period housing and mortgage is zero
            UC_prime_func1D = beta * UC_prime_func_ex[DB_ind,
                                                      E_ind, alpha_ind,
                                                      beta_ind, Pi_ind, :,
                                                      adc_ind, 0, q_ind, 0]

            # calculate optimal consumption for h and q value
            c_t = max(C_min, phi_r * h * q *
                      (1 - alpha_housing) / alpha_housing)

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

            a_end_clean = np.sort(a_end_4[i][a_end_4[i]>0])
            hs_sorted = np.take(H_R[a_end_4[i]>0],
                                np.argsort(a_end_4[i][a_end_4[i]>0]))

            if len(a_end_clean)== 0:
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

    @njit(parallel=True, nogil=True)
    def interp_pol_noadj(assets_l, etas, cons_l):
        """ Interpolates no adjust policy functions using
        from endogenous grids"""

        assets_reshaped = assets_l.reshape(all_state_shape)
        assets_reshaped = np.transpose(assets_reshaped,
                                       (0, 1, 2, 3, 4, 6, 7, 8, 9, 5))
        assets_reshaped_1 = np.copy(assets_reshaped)\
            .reshape((int(len(X_all_hat_vals)
                          / grid_size_A), int(grid_size_A)))

        etas_reshaped = etas.reshape(all_state_shape)
        etas_reshaped = np.transpose(np.copy(etas_reshaped),
                                     (0, 1, 2, 3, 4, 6, 7, 8, 9, 5))
        etas_reshaped_1 = np.copy(etas_reshaped)\
            .reshape((int(len(X_all_hat_vals)
                          / grid_size_A),
                      int(grid_size_A)))

        cons_reshaped = cons_l.reshape(all_state_shape)
        cons_reshaped = np.transpose(cons_reshaped,
                                     (0, 1, 2, 3, 4, 6, 7, 8, 9, 5))
        cons_reshaped_1 = np.copy(cons_reshaped)\
            .reshape((int(len(X_all_hat_vals)
                          / grid_size_A), int(grid_size_A)))

        # generate empty grids to fill with interpolated functions
        Aprime_noadjust_1 = np.zeros((int(len(X_all_hat_vals)
                                          / grid_size_A), int(grid_size_A)))
        etas_primes_1 = np.zeros((int(len(X_all_hat_vals)
                                      / grid_size_A),
                                  int(grid_size_A)))
        C_noadj_1 = np.zeros((int(len(X_all_hat_vals)
                                  / grid_size_A), int(grid_size_A)))

        # Interpolate
        for i in prange(len(assets_reshaped_1)):

            a_prime_points = np.take(A[~np.isnan(assets_reshaped_1[i])], np.argsort(assets_reshaped_1[i]))
            
            Aprime_noadjust_1[i] = interp_as(assets_reshaped_1[i],
                                             a_prime_points,
                                             A)
            c_prime_points = np.take(cons_reshaped_1[i][~np.isnan(assets_reshaped_1[i])],
                                     np.argsort(assets_reshaped_1[i]))

            
            C_noadj_1[i] = interp_as(assets_reshaped_1[i],
                                     c_prime_points,
                                     A)

            Aprime_noadjust_1[i][Aprime_noadjust_1[i] < 0] = A_min
            eta_prime_points = np.take(etas_reshaped_1[i][~np.isnan(assets_reshaped_1[i])],
                                       np.argsort(assets_reshaped_1[i]))

            etas_primes_1[i] = interp_as(assets_reshaped_1[i],
                                         eta_prime_points,
                                         A)

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
                       wealth_bar):

        """ Interpolates no adjust policy functions using
        from endogenous grids"""

        A_prime_adj_reshape = reshape_make_h_last(A_prime)
        C_adj = reshape_make_h_last(C)
        wealth_bar_reshape = reshape_make_h_last(wealth_bar)

        # New pol funcs which will be interpolated over uniform
        # wealth grid
        assets_prime_adj_1 \
            = np.zeros((int(len(X_W_bar_hdjex_ind) / grid_size_HS),
                        int(grid_size_A)))
        H_prime_adj_1\
            = np.zeros((int(len(X_W_bar_hdjex_ind) / grid_size_HS),
                        int(grid_size_A)))
        c_prime_adj_1\
            = np.zeros((int(len(X_W_bar_hdjex_ind) / grid_size_HS),
                        int(grid_size_A)))

        # interpolate over uniform wealth grid (1D)
        for i in prange(len(wealth_bar_reshape)):

            wealth_x = wealth_bar_reshape[i][~np.isnan(wealth_bar_reshape[i])]
            assets_clean = A_prime_adj_reshape[i][~np.isnan(wealth_bar_reshape[i])]
            assts_x = np.take(assets_clean, np.argsort(wealth_x))

            h_clean = H_R[~np.isnan(wealth_bar_reshape[i])]
            h_x = np.take(h_clean, np.argsort(wealth_x))

            #print(h_x)
            c_clean = C_adj[i][~np.isnan(wealth_bar_reshape[i])]
            c_x = np.take(c_clean, np.argsort(wealth_x))

            wealth_xs = np.sort(wealth_x)

            assets_prime_adj_1[i] = interp_as(wealth_xs, assts_x, W_W)
            assets_prime_adj_1[i][assets_prime_adj_1[i] <= 0] = A_min

            c_prime_adj_1[i] = interp_as(wealth_xs, c_x, W_W)
            c_prime_adj_1[i][c_prime_adj_1[i] <= 0] = C_min

            H_prime_adj_1[i] = interp_as(wealth_xs, h_x, W_W)
            H_prime_adj_1[i][H_prime_adj_1[i] <= 0] = H_min

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

        return C_adj, H_adj, Aprime_adj

    @njit(parallel=True, nogil= True)
    def eval_policy_W_noadj(t,
                            mort_func,
                            UC_prime_func,
                            UC_prime_H_func,
                            UC_prime_HFC_func,
                            UC_prime_M_func,
                            UF_prime_func):
        """Time t worker policy function evaluation

        Parameters
        ----------
        t : int
             age
        mort_func: 9D array
            M_t+1 function
        UC_prime_func : flat 9D array
                t+1 expected value of u_1(c,h)
        UC_prime_H_func: flat 9D array
                t+1 RHS of Euler equation wrt to H_t
        UC_prime_HFC_func:  flat 9D array
                t+1 RHS fixed cost of adj
        UC_prime_M_func: flat 9D array
                t+1 RHS of Euler equation wrt to M_t
        UF_prime_func: flat 9D array
                t+1 utility

        Returns
        -------
        C_noadj: 9D array
            no adjust a_t+1 policy
        etas_primes: 9D array
            no adjust multiplier (equation x)
        assets_prime_adj: 8D array
            adjust policy
        C_prime_adj: 8D array
            adjust policy for consumption
        H_prime_adj: 8D array
            adjust policy for housing


        Notes
        -----
        Expectation of UC_prime_func and UC_prime_eta_func defined on:

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

        How doea min_const_FOC work when housing is adjusted down?
        """

        # gen empty grids to fill with
        # endogenous values of assets_l (liq assets inc. wage)
        # eta and consumption

        assets_l = np.zeros(len(X_all_hat_vals))
        etas = np.zeros(len(X_all_hat_vals))
        cons_l = np.zeros(len(X_all_hat_vals))

        # Reshape UC functions so they can be indexed by current
        # period states and be made into a function of M_t+1
        UC_prime_func_ex = UC_prime_func.reshape(all_state_shape)
        UC_prime_H_func_ex = UC_prime_H_func.reshape(all_state_shape)
        UC_prime_HFC_func_ex = UC_prime_HFC_func.reshape(all_state_shape)
        UC_prime_M_func_ex = UC_prime_M_func.reshape(all_state_shape)
        UF_prime_func_ex = UF_prime_func.reshape(all_state_shape)

        for i in prange(len(X_all_hat_vals)):
            # Loop through each exogenous grid point
            # recall h is H_t i.e. housing *after* depreciation at t
            # i.e. housing that goes into utility at t
            # a_prime is liquid asset taken into next period
            # q is period t price

            h = X_all_hat_vals[i][7] * (1 - delta_housing)
            a_prime = X_all_hat_vals[i][5]
            q = X_all_hat_vals[i][8]
            m = X_all_hat_vals[i][9]
            a_dc = X_all_hat_vals[i][6]

            E_ind = X_all_hat_ind[i][1]
            alpha_ind = X_all_hat_ind[i][2]
            beta_ind = X_all_hat_ind[i][3]
            Pi_ind = X_all_hat_ind[i][4]
            DB_ind = 0
            a_ind = int(X_all_hat_ind[i][5])
            adc_ind = int(X_all_hat_ind[i][6])
            h_ind = int(X_all_hat_ind[i][7])
            q_ind = int(X_all_hat_ind[i][8])

            beta = np.exp(X_all_hat_vals[i][3]
                          + np.log(beta_bar))
            alpha_housing = np.exp(X_all_hat_vals[i][2]
                                   + np.log(alpha_bar))

            # get the next period U_prime values
            mfunc_ucprime = UC_prime_func_ex[DB_ind,
                                             E_ind, alpha_ind,
                                             beta_ind, Pi_ind,
                                             :, adc_ind, h_ind,
                                             q_ind, :]
            mfunc_uc_h_prime = UC_prime_H_func_ex[DB_ind,
                                                  E_ind, alpha_ind,
                                                  beta_ind, Pi_ind,
                                                  :, adc_ind, h_ind,
                                                  q_ind, :]
            mfunc_uc_hfc_prime = UC_prime_HFC_func_ex[DB_ind,
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
            uf_prime_func = UF_prime_func_ex[DB_ind, E_ind,
                                             alpha_ind,
                                             beta_ind, Pi_ind,
                                             :, adc_ind, h_ind,
                                             q_ind]

            # calculate consumption by checking whether
            # mortgage liability binds at any constraints

            # RHS with min payment
            min_pay_points = np.array([a_prime,\
                                    (1 - amort_rate(t - 2)) * m])
            ucmprime_m_minp = max(1e-200,
                                  beta * eval_linear(X_cont_WAM,
                                                     mfunc_uc_m_prime,
                                                     min_pay_points,
                                                     xto.LINEAR))
            uc_prime_minp = max(1e-200,
                                beta * eval_linear(X_cont_WAM,
                                                   mfunc_ucprime,
                                                   min_pay_points,
                                                   xto.LINEAR))

            # RHS with full payment
            max_pay_points = np.array([a_prime, 0])
            ucmprime_m_fp = max(1e-200, beta
                                * eval_linear(X_cont_WAM,
                                              mfunc_uc_m_prime,
                                              max_pay_points,
                                              xto.LINEAR))
            uc_prime_fp = max(1e-200, beta
                              * eval_linear(X_cont_WAM,
                                            mfunc_ucprime,
                                            max_pay_points,
                                            xto.LINEAR))

            # Step 1: check if constrained by min payment
            if m == 0:
                c_t = min(C_max,max(C_min, uc_inv(uc_prime_fp, h, alpha_housing)))
                m_prime = 0
                UC_prime_RHS = uc_prime_fp

            elif uc_prime_minp >= ucmprime_m_minp:
                c_t = min(C_max,max(C_min,uc_inv(uc_prime_minp, h, alpha_housing)))
                m_prime = (1 - amort_rate(t - 2)) * m
                UC_prime_RHS = uc_prime_minp

            # Step 2: check if constrainte by full payment
            elif uc_prime_fp <= ucmprime_m_fp:
                c_t = min(max(C_min, uc_inv(uc_prime_fp, h, alpha_housing)), C_max)
                m_prime = 0
                UC_prime_RHS = uc_prime_fp

            # Step 3: otherwise, eval unconstrained:
            else:
                m_prime_adj = interp_as(A, mort_func1D, np.array([a_prime]), extrap= True)[0]
                m_prime = max(0, min(m_prime_adj, (1 - amort_rate(t - 2)) * m))
                UC_prime_RHS = max(1e-250,
                                   beta * eval_linear(X_cont_WAM,
                                                      mfunc_ucprime,
                                                      np.array([a_prime, m_prime]),
                                                      xto.LINEAR))
                c_t = min(C_max,max(C_min, uc_inv(UC_prime_RHS, h, alpha_housing)))

            extra_payment = max(0, (1 - amort_rate(t - 2)) * m - m_prime)
            min_constrained = extra_payment == 0\
                and (1 - amort_rate(t - 2)) * m <= h * q * (1 - phi_c)

            # Constraint that tells us that mortgage liability is
            # greater than house value (after adjustment, agent will
            # have to pay excess liability)
            ref_max_constrained = m_prime > h * q * (1 - phi_c)
            pts_noadj = np.array([a_prime, m_prime])
            if ref_max_constrained == 1:
                # calculate the value of lost utility from discrete
                # jump in payment of mortgage if forced to re-finance
                # when housing is adjusted (if mort> house val*(1-phi))
                # after forced refinancing, house val*(1-phi)> mort
                pts_atrefc = np.array([a_prime, h * q * (1 - phi_c)])
                prime_diff_val = beta * eval_linear(X_cont_WAM,
                                                    uf_prime_func,
                                                    pts_atrefc,
                                                    xto.LINEAR)\
                                         - beta * eval_linear(X_cont_WAM,
                                                             uf_prime_func,
                                                             pts_noadj,
                                                             xto.LINEAR)

                # Calculate what consumption would be after forced
                # refinance
                uc_prime_refc = max(1e-200,
                                    beta * eval_linear(X_cont_WAM,
                                                       mfunc_ucprime,
                                                       pts_atrefc,
                                                       xto.LINEAR))
                c_t_at_refconst = min(C_min,uc_inv(uc_prime_refc, h, alpha_housing))
                curr_diff = u(c_t_at_refconst, h, alpha_housing)\
                    - u(c_t, h, alpha_housing)

                val_diff_refc = curr_diff + prime_diff_val

            # calculate values of RHS of Euler

            UC_prime_H_RHS = beta * eval_linear(X_cont_WAM,
                                                mfunc_uc_h_prime,
                                                pts_noadj,
                                                xto.LINEAR)
            UC_prime_HFC_RHS = beta * eval_linear(X_cont_WAM,
                                                  mfunc_uc_hfc_prime,
                                                  pts_noadj,
                                                  xto.LINEAR)
            UC_prime_M_RHS = beta * eval_linear(X_cont_WAM,
                                                mfunc_uc_m_prime,
                                                pts_noadj,
                                                xto.LINEAR)

            # Define the endogenous liq assets after
            # wage and after vol. cont. and after making
            # min mort. payment
            # note we are NOT dividing through by (1+r)
            assets_l[i] = c_t + a_prime + extra_payment
            cons_l[i] = c_t

            # Calculate the no-adjust multipler
            # min_const_FOC is marginal benefit from relaxing collat
            # constraint by adjusting increasing house value
            min_const_FOC = min_constrained\
                * (max(uc(c_t, h, alpha_housing) - ucmprime_m_minp, 0))

            ref_const_FOC = -ref_max_constrained * (min(val_diff_refc, 0))
            
            geta_t = uh(c_t, h, alpha_housing)\
                + min_const_FOC\
                + UC_prime_H_RHS\
                - uc(c_t, h, alpha_housing) * q
            
            etas[i] = geta_t / (uc(c_t, h, alpha_housing)
                                * q * h * tau_housing
                                + UC_prime_HFC_RHS + ref_const_FOC)

        # Re-shape and interpolate the no adjustment endogenous grids
        # we want rows to  index a product of states
        # other than A_prime.
        # vals in the columns are time t liquid asssets for A_t+1
        C_noadj, etas_noadj, Aprime_noadj\
            = interp_pol_noadj(assets_l, etas, cons_l)

        return C_noadj, etas_noadj, Aprime_noadj

    @njit
    def eval_policy_W_adj(t, mort_func,
                          UC_prime_func,
                          UC_prime_H_func,
                          UC_prime_HFC_func,
                          UC_prime_M_func):
        """Time t worker policy function evaluation
                with adjustment

        Parameters
        ----------
        t :                 int
                                                 age
        UC_prime_func :     flat 10D array
                                                 t+1 expected value of u_1(c,h)
        UC_prime_H_func:    flat 10D array
                                                 t+1 RHS of Euler equation wrt to H_t
        UC_prime_HFC_func:  flat 10D array
                                                 t+1 RHS fixed cost of adj
        UC_prime_M_func:    flat 10D array
                                                 t+1 mortgage choice

        Returns
        -------
        C_adj:              9D array
                                                 adjuster consumption policy
        H_adj:              9D array
                                                 adjust housing policy
        Aprime_adj:         9D array
                                                 adjust liquid assett policy

        Expectation of UC_prime_func and UC_prime_eta_func defined on:

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

        UC_prime_func_ex = UC_prime_func.reshape(all_state_shape)
        UC_prime_H_func_ex = UC_prime_H_func.reshape(all_state_shape)
        UC_prime_HFC_func_ex = UC_prime_HFC_func.reshape(all_state_shape)
        UC_prime_M_func_ex = UC_prime_M_func.reshape(all_state_shape)

        for i in prange(len(X_W_bar_hdjex_ind)):

            """
            For each i, the indices in
            X_W_bar_hdjex_ind[i] are:

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
            h, q, a_dc, m = X_W_bar_hdjex_vals[i][6],\
                X_W_bar_hdjex_vals[i][7],\
                X_W_bar_hdjex_vals[i][5],\
                X_W_bar_hdjex_vals[i][8]

            E_ind, alpha_ind, beta_ind = X_W_bar_hdjex_ind[i][1],\
                X_W_bar_hdjex_ind[i][2],\
                X_W_bar_hdjex_ind[i][3]
            Pi_ind, DB_ind, adc_ind = X_W_bar_hdjex_ind[i][4],\
                0,\
                int(X_W_bar_hdjex_ind[i][5])
            h_ind, q_ind = int(X_W_bar_hdjex_ind[i][6]),\
                int(X_W_bar_hdjex_ind[i][7])

            beta = np.exp(X_W_bar_hdjex_vals[i][3]
                          + np.log(beta_bar))
            alpha_housing = np.exp(X_W_bar_hdjex_vals[i][2]
                                   + np.log(alpha_bar))

            # get functions for next period U_prime values
            # as functions of A_prime and M_prime
            UC_prime_func2D = UC_prime_func_ex[DB_ind,
                                               E_ind, alpha_ind,
                                               beta_ind, Pi_ind,
                                               :, adc_ind, h_ind,
                                               q_ind, :]

            UC_prime_H_func2D = UC_prime_H_func_ex[DB_ind,
                                                   E_ind, alpha_ind,
                                                   beta_ind, Pi_ind,
                                                   :, adc_ind, h_ind,
                                                   q_ind, :]

            UC_prime_HFC_func2D = UC_prime_HFC_func_ex[DB_ind,
                                                       E_ind, alpha_ind,
                                                       beta_ind, Pi_ind,
                                                       :, adc_ind, h_ind,
                                                       q_ind, :]

            UC_prime_M_func2D = UC_prime_M_func_ex[DB_ind,
                                                   E_ind, alpha_ind,
                                                   beta_ind, Pi_ind,
                                                   :, adc_ind, h_ind,
                                                   q_ind, :]
            # get function of M_prime
            # as function of A_prime
            mort_func1D = mort_func[DB_ind, E_ind,
                                    alpha_ind,
                                    beta_ind, Pi_ind,
                                    :, adc_ind, h_ind,
                                    q_ind]
            # as function of A_prime and H_t
            mort_func2D = mort_func[DB_ind, E_ind,
                                    alpha_ind,
                                    beta_ind, Pi_ind,
                                    :, adc_ind, :,
                                    q_ind]
            # Calc wage (for max mortgage)
            wage = y(t, E[E_ind])

            # make tuple of args for FOC Euler error functions
            args_HA_FOC = (a_dc, h, q, m, alpha_housing, beta,
                           wage, mort_func1D,
                           UC_prime_func2D, UC_prime_H_func2D,
                           UC_prime_HFC_func2D, UC_prime_M_func2D, t)

            args_eval_c = (A_min, a_dc, h, q, m, alpha_housing, beta,
                           wage, mort_func1D,
                           UC_prime_func2D, UC_prime_H_func2D,
                           UC_prime_HFC_func2D, UC_prime_M_func2D, t)

            A_prime[i] = np.nan
            C[i] = np.nan
            wealth_bar[i] = np.nan
            max_loan =  min(q * h * (1 - phi_c),
                                      phi_d * wage / amort_rate(t - 1))

            # check if there is an interior solution to
            # housing, liq assett and constrained mortgage FOC
            if HA_FOC(A_min, *args_HA_FOC)\
                    * HA_FOC(A_max_W, *args_HA_FOC) < 0:
                # if interior solution to a_t+1, calculate it
                A_prime[i] = max(brentq(HA_FOC, A_min, A_max_W,
                                        args=args_HA_FOC, 
                                            xtol= 1e-05)[0],
                                        A_min)

                C[i] = min(C_max,max(
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
                        UC_prime_HFC_func2D,
                        UC_prime_M_func2D,
                        t,
                        ret_cons=True)))
                m_prime_adj = interp_as(A, mort_func1D,
                                        np.array([A_prime[i]]),
                                        extrap = True)[0]
                m_prime = max(0, min(m_prime_adj, max_loan))
                extra_payment = (1 - amort_rate(t - 2)) * m - m_prime

                # end. cash at hand wealth is defined as:
                # (1+r)a_t + net wage - minimum mortage payment
                # + P_t(1-delta)H_t-1

                wealth_bar[i] = C[i] + A_prime[i]\
                    + q * h * (1 + tau_housing)\
                    + extra_payment

            # if no interior solution with liq asset unconstrainted,
            # check if interior solution to housing and constrained
            # mortage with binding liquid assett FOC at A_min
            elif H_FOC(C_min, *args_eval_c)\
                    * H_FOC(C_max, *args_eval_c) < 0:
                C_at_amin = min(C_max,max(brentq(H_FOC, C_min, C_max,
                                       args=args_eval_c, xtol = 1e-05)[0], C_min))

                m_prime_adj_at_amin1 = interp_as(A, mort_func1D,
                                                np.array([A_min]),extrap = True)[0]
                m_prime_at_amin = min(m_prime_adj_at_amin1, max_loan)
                UC_prime_RHS_amin = beta\
                    * eval_linear(X_cont_WAM,
                                  UC_prime_func2D,
                                  np.array([A_min, m_prime_at_amin]),
                                  xto.LINEAR)

                # if liquid assett const. does not satisfy
                # FOC, throw point out
                if uc(C_at_amin, h, alpha_housing) >= beta * UC_prime_RHS_amin:
                    A_prime[i] = A_min
                    C[i] = C_at_amin
                    extra_payment = (1 - amort_rate(t - 2)) * m - m_prime_at_amin
                    wealth_bar[i] = C[i] \
                        + A_prime[i]\
                        + q * h * (1 + tau_housing)\
                        + extra_payment

            # force include point with zero housing
            # if solution adjusted to housing
            # less than H_min
            elif h == H_min:
                A_prime[i] = A_min
                C[i] = C_min
                wealth_bar[i] = C_min + A_min + q * h * (1 + tau_housing)

        # interpolate C, H and A_prime on endogenous wealth grid

        C_adj, H_adj, Aprime_adj = interp_pol_adj(A_prime, C, wealth_bar)

        return C_adj, H_adj, Aprime_adj

    # Functions to generate and condition the RHS of Euler equation

    @njit
    def gen_RHS_pol_adj(adj_pols, points_adj):
        """ Evalutes policies for housing adjusters on all points in the
                full state-space grid"""

        # Empty adjuster policy grid
        adj_vals_small1 = np.empty((len(adj_pols),
                                    int(grid_size_H * len(V)
                                        * grid_size_A
                                        * grid_size_DC), 5))

        for i in prange(len(adj_pols)):
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
            adj_vals_small1[i, :, 0:3] = eval_linear(X_QH_W, adj_pols[i],
                                                     points_adj[i],
                                                     xto.LINEAR)

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

        # Reshape the evaluated points to X_all_ind.shape
        adj_vals_small2 = np.copy(adj_vals_small1)\
            .reshape((1, grid_size_W,
                      grid_size_alpha,
                      grid_size_beta, len(Pi),
                      grid_size_Q, grid_size_M,
                      len(V),
                      grid_size_A, grid_size_DC,
                      grid_size_H, 5))
        adj_vals_small3 = adj_vals_small2.\
            transpose((0, 1, 2, 3, 7, 4, 8, 9, 10, 5, 6, 11))
        adj_vals = np.copy(adj_vals_small3).reshape((len(X_all_ind), 5))

        return adj_vals

    @njit
    def gen_RHS_pol_noadj(noadj_pols, points_noadj):
        """ Evalutes pol for housing non-adjusters on all points in the
                full state-space grid"""

        # Empty no-adjuster policy array
        noadj_vals_small1 = np.empty((len(noadj_pols),
                                      int(len(V) * grid_size_A
                                          * grid_size_DC),
                                      4))

        for i in prange(len(noadj_pols)):
            noadj_vals_small1[i, :, 0:3] = eval_linear(X_cont_W_hat,
                                                       noadj_pols[i],
                                                       points_noadj[i],
                                                       xto.LINEAR)

            noadj_vals_small1[i, :, 0][noadj_vals_small1[i, :, 0]<=0] = C_min 
            #noadj_vals_small1[i, :, 1][noadj_vals_small1[i, :, 1]<=0] = A_min
            noadj_vals_small1[i, :, 2][noadj_vals_small1[i, :, 2]<=0] = A_min

            noadj_vals_small1[i, :, 3] = points_noadj[i, :, 0]\
                - noadj_vals_small1[i, :, 0]\
                - noadj_vals_small1[i, :, 2]

            # Recall for non-adjusters, extra-pay cannot be less than 0
            noadj_vals_small1[i, :, 3][noadj_vals_small1[i, :, 3]<=0]= 0

        noadj_vals_small2 = np.copy(noadj_vals_small1).reshape(
            (1,
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
                4))

        noadj_vals_small3 = noadj_vals_small2\
            .transpose((0, 1, 2, 3, 8, 4, 9, 10, 5, 6, 7, 11))
        noadj_vals = np.copy(noadj_vals_small3)\
            .reshape((len(X_all_ind), 4))

        return noadj_vals

    @njit
    def gen_RHS_pol_rent(H_rent, points_rent):
        """ Evalutes pol for housing renters on all points in the
                full state-space grid"""

        # Empty array for renter pols
        rent_vals_small1 = np.empty((len(H_rent),
                                     int(grid_size_H
                                         * grid_size_M
                                         * len(V)
                                         * grid_size_A
                                         * grid_size_DC),
                                     2))
        
        for i in prange(len(H_rent)):
            rent_vals_small1[i, :, 0] = eval_linear(X_QH_W,
                                                    H_rent[i],
                                                    points_rent[i], xto.LINEAR)
            rent_vals_small1[i, :, 1] = points_rent[i, :, 0]

        rent_vals_small2 = np.copy(rent_vals_small1)\
                                .reshape((1,
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
                                            2))

        # Reshape the evaluated points to X_all_ind.shape
        rent_vals_small3 = rent_vals_small2.\
            transpose((0, 1, 2, 3, 6, 4, 7, 8, 9, 5, 10, 11))
        rent_vals = np.copy(rent_vals_small3).\
            reshape((len(X_all_ind), 2))

        return rent_vals

    @njit
    def gen_RHS_pol_val(noadj_pols,
                        adj_pols,
                        H_rent,
                        points_noadj,
                        points_adj,
                        points_rent):
        """ Function evaluates policy and value functions
                 for non-adjusters, adjusters and renters
                 on X_all_ind grid"""

        # RHS policy for housing adjusters ,non-adjusters and renters

        adj_vals, noadj_vals, rent_vals\
            = gen_RHS_pol_adj(adj_pols, points_adj),\
            gen_RHS_pol_noadj(noadj_pols, points_noadj),\
            gen_RHS_pol_rent(H_rent, points_rent)

        # Generate indicator function for non-adjuster (|eta|<=1)
        eta_vals = np.abs(noadj_vals[:, 1])
        eta_vals[np.isnan(eta_vals)] = 0
        etas_ind = np.where(eta_vals <= 1, 1, 0)

        # Generate policies for non-renters
        # eta*no-adjust + (1-eta)*adjust
        # recall final 5th col of adj_vals is
        # A_DC prime vals (common for renters, adjusters \
        # and non-adjusters since it only depends on V_t
        h_prime_norent_vals = (1 - etas_ind) * adj_vals[:, 1]\
            + etas_ind * H[X_all_ind[:, 8]]
        h_prime_norent_vals[h_prime_norent_vals<=H_min]= H_min
        
        c_prime_norent_vals = (1 - etas_ind)*np.copy(adj_vals[:, 0]) + etas_ind*np.copy(noadj_vals[:, 0])
        c_prime_norent_vals[np.isnan(c_prime_norent_vals)] = C_min
        c_prime_norent_vals[c_prime_norent_vals< C_min]= C_min
        c_prime_norent_vals[c_prime_norent_vals>C_max]= C_max
        
        a_prime_norent_vals = (1 - etas_ind) * adj_vals[:, 2]\
            + etas_ind * noadj_vals[:, 2]
        a_prime_norent_vals[a_prime_norent_vals<=A_min] = A_min
        extra_pay = (1 - etas_ind) * adj_vals[:, 3]\
            + etas_ind * noadj_vals[:, 3]
        adcprime = adj_vals[:, 4]
        extra_pay[np.isnan(extra_pay)]= 0

        # Renter polices
        h_prime_rent = rent_vals[:, 0]
        h_prime_rent[h_prime_rent<= H_min] = H_min
        c_prime_rent = phi_r * Q[X_all_ind[:, 9]] * h_prime_rent\
            * (1 - X_all_ind_W_vals[:, 0]) / X_all_ind_W_vals[:, 0]
        c_prime_rent[c_prime_rent <= C_min] = C_min

        # Recall 2nd col. of rent vals is the cash at hand for renters
        # at t, thus A_t+1 = cash at hand - rent payments - consumption
        a_prime_rent = rent_vals[:, 1] - c_prime_rent \
            - h_prime_rent * phi_r * Q[X_all_ind[:, 9]]
        a_prime_rent[a_prime_rent<=A_min] = A_min

        # Cannot Switch to renting if cash at hand is zero after making
        # full mortgage payment
        cannot_rent = rent_vals[:, 1] == A_min

        # Marginal utilis for renters and non-renters
        ucfunc_norent = uc_vec(c_prime_norent_vals,
                               h_prime_norent_vals, X_all_ind_W_vals[:, 0])
        ufunc_norent = u_vec(c_prime_norent_vals,
                             h_prime_norent_vals, X_all_ind_W_vals[:, 0])
        ucfunc_rent = uc_vec(c_prime_rent,
                             h_prime_rent, X_all_ind_W_vals[:, 0])
        ufunc_rent = u_vec(c_prime_rent,
                           h_prime_rent, X_all_ind_W_vals[:, 0])
        #print(np.sum(np.isnan(ufunc_rent)))

        return c_prime_norent_vals, h_prime_norent_vals, a_prime_norent_vals,\
            ucfunc_norent, ufunc_norent, extra_pay, etas_ind,\
            cannot_rent, c_prime_rent, h_prime_rent, a_prime_rent,\
            ucfunc_rent, ufunc_rent, adcprime

    @njit
    def gen_RHS_lambdas(t, V_funcs,
                        UF_B_rent,
                        UF_B_norent,
                        no_rent_points_p,
                        rent_points_p):
        """ Generates RHS values of VF_t+1, Lambda_t+1
                 and Xi_t for renters and non-renters
                 conditioned on time t states in X_all_ind"""

        # Generate empty arrays to fill with evaluated values
        Xi_rent \
            = np.empty((len(no_rent_points_p), len(X_V_func_DP_vals[:, 0])))
        Xi_norent = np.copy(Xi_rent)
        Lambda_B_rent = np.copy(Xi_rent)
        VF_B_rent = np.copy(Xi_rent)
        Lambda_B_norent = np.copy(Xi_rent)
        VF_B_norent = np.copy(Xi_rent)
        zeta = np.copy(Xi_rent)

        # The list of vol contributions rates will be common across each
        # i in the loop below
        v = X_V_func_DP_vals[:, 0]

        for i in prange(len(no_rent_points_p)):

            """
            Each i of no_rent_points_p is a
            list of t+1 policies, indexed by
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
            beta_t = np.exp(X_V_func_CR_vals[i, 3]
                            + np.log(beta_bar))

            # Interpolate no renter value function
            # returns list of |V_txA_txA_DC_txH_txM_t|x2 points
            # first column is V_t+1
            val_func_vals_norent = beta_t * eval_linear(X_cont_W_bar,
                                                        V_funcs[i],
                                                        no_rent_points_p[i],
                                                        xto.LINEAR)

            #print(np.sum(np.isnan(val_func_vals_norent)))

            # check: should bequest value go here?
            VF_B_norent[i, :] = UF_B_norent[i, :] + val_func_vals_norent[:, 0]

            Lambda_B_norent[i, :] = val_func_vals_norent[:, 1]
            Xi_norent[i] = UF_B_norent[i, :]\
                + Lambda_B_norent[i, :] * no_rent_points_p[i, :, 1]\
                - adj_v(t, X_V_func_DP_vals[:, 1]) * (v > 0)\
                - adj_pi(t, X_V_func_DP_vals[:, 2],
                         adj_p(17)) * (pi != def_pi)

            # calculate renter value functions
            val_func_vals_rent = beta_t * eval_linear(X_cont_W_bar,
                                                      V_funcs[i],
                                                      rent_points_p[i],
                                                      xto.LINEAR)
            #print(np.sum(np.isnan(val_func_vals_rent)))

            Lambda_B_rent[i, :] = val_func_vals_rent[:, 1]
            VF_B_rent[i, :] = UF_B_rent[i, :]\
                + val_func_vals_rent[:, 0]
            Xi_rent[i, :] = UF_B_rent[i, :]\
                + Lambda_B_rent[i, :] * (rent_points_p[i, :, 1])\
                - adj_v(t, X_V_func_DP_vals[:, 1]) * (v > 0)\
                - adj_pi(t, X_V_func_DP_vals[:, 2],
                         adj_p(17)) * (pi != def_pi)

            # Multiplier to determine renter or no renter
            zeta[i, :] = VF_B_rent[i, :] - VF_B_norent[i, :]

        # Rehsape all to X_all_ind.shape() and return
        return reshape_RHS_Vfunc_rev(Xi_rent),\
            reshape_RHS_Vfunc_rev(Xi_norent),\
            reshape_RHS_Vfunc_rev(Lambda_B_rent),\
            reshape_RHS_Vfunc_rev(Lambda_B_norent),\
            reshape_RHS_Vfunc_rev(VF_B_rent),\
            reshape_RHS_Vfunc_rev(VF_B_norent),\
            reshape_RHS_Vfunc_rev(zeta)

    @njit
    def gen_RHS_UC_func(t, noadj_pols,
                        adj_pols,
                        H_rent,
                        V_funcs,
                        points_noadj_vec,
                        points_adj_vec,
                        points_rent_vec,
                        A_prime):
        """
        Generate the unconditioned marginal utilities on a
                grid for time t

                The marginal utilities hese will be *integrands*
                in the conditional expec. of the RHS of the
                Euler equation at t-1

                note the marginal utilities are defined on
                the state conditioned on the discrete
                pension choices

        Parameters
        ----------
        t:          int
                                 age
        Aprime_noadj:    flat 9D array
                                 t+1 expected value of u_1(c,h)

        etas_noadj: flat 9D array
                                 t+1 RHS of Euler equation wrt to H_t
        Aprime_adj:  flat 9D array
                                         t+1 RHS fixed cost of adj
        C_adj:  flat 9D array
                                         t+1 RHS fixed cost of adj
        Lambda_prime:   flat 9D array

        Returns
        ----------
        UC_prime_B:        flat 9D array
        UC_prime_H_B:      flat 9D array
        UC_prime_HFC_B:    flat 9D array
        Lambda_B:          flat 9D array
        Xi:                flat 9D array

        """

        # Generate the time t policies interpolated
        # on X_all_ind grid
        c_prime_norent_vals, h_prime_norent_vals, a_prime_norent_vals,\
            uc_prime_norent, u_prime_norent, extra_pay, etas_ind,\
            cannot_rent, c_prime_rent_vals, h_prime_rent_vals, a_prime_rent_vals,\
            uc_prime_rent, u_prime_rent, adcprime\
            = gen_RHS_pol_val(noadj_pols,
                              adj_pols, H_rent,
                              points_noadj_vec,
                              points_adj_vec,
                              points_rent_vec)

        # Extra_pay is vector of mortgage payments in addition
        # the min. amort payment
        # Recall that all renters are not min payment
        # constrained as they have to pay off thier mortgage
        extra_pay[extra_pay < 0] = 0
        extra_pay_ind1 = extra_pay > 1e-5

        # t+1 period mortgage balance for non-renters
        m_prime = (1 - amort_rate(t - 2)) * M[X_all_ind[:, 10]]\
            - extra_pay
        q_vec = Q[X_all_ind[:, 9]]
        m_prime[m_prime < 0] = 0

        # time t utility vector for  all X_all_ind states
        UF_B_norent = u_prime_norent
        UF_B_rent = u_prime_rent


        # Generate t+1 continuous states given policies
        # these are used to evaluate tiem t conditional value function
        # and Lamba function
        no_rent_points_p = reshape_vfunc_points(np.column_stack(
            (a_prime_norent_vals, adcprime, h_prime_norent_vals, m_prime)))

        rent_points_p = reshape_vfunc_points(np.column_stack((
            a_prime_rent_vals,
            adcprime,
            np.full(len(X_all_ind), H_min),
            np.full(len(X_all_ind), 0))))

        # Loop over all points of the states in  X_all_ind
        # interpolate values of value function and
        # Lambda function back to period t
        Xi_rent, Xi_norent, Lambda_B_rent, Lambda_B_norent, VF_B_rent,\
            VF_B_norent, zeta = gen_RHS_lambdas(t, V_funcs,
                                                reshape_RHS_UFB(UF_B_rent),
                                                reshape_RHS_UFB(UF_B_norent),
                                                no_rent_points_p,
                                                rent_points_p)


        # zeta is multiplier that indicates renter if zeta>1
        # cannot rent is an indictator that someone is not able
        # to pay off thier mortgage liability after renting decision
        # hence will not rent 
        # renter_ind = zeta>0 AND cannot_rent == 0 
        zeta[np.isnan(zeta)] = 0
        renter_ind1 = zeta > 0
        renter_ind = renter_ind1*(1-cannot_rent)

        renter_ind[np.isnan(renter_ind)] =0

        # Recall that all renters are not bounded by min payment
        # hence thier FOC is the extra_pay FOC
        extra_pay_ind = extra_pay_ind1 * (1 - renter_ind)\
            + renter_ind
        uc_prime = renter_ind * u_prime_rent\
            + (1 - renter_ind) * uc_prime_norent

        # RHS values of Euler equation
        # First, with respect to A_t
        UC_prime_B = (1 + r) * (s[int(t - 1)] * uc_prime + (1 - s[int(t - 1)])
                                * b_prime(A_prime))
        # wrt to H_t-1
        UC_prime_H_B = q_vec\
            * (1 - delta_housing - tau_housing * renter_ind)\
            * (s[int(t - 1)] * uc_prime
               + (1 - s[int(t - 1)]) * b_prime(A_prime))

        # the fixed cost paif if H_t-1 is adjusted
        UC_prime_HFC_B = q_vec * (1 - delta_housing) * s[int(t - 1)] * uc_prime\
            * tau_housing * h_prime_norent_vals\
            * etas_ind * (1 - renter_ind)
        # wrt to mortgage m_t
        UC_prime_M_B = s[int(t - 1)] * uc_prime * extra_pay_ind\
            + (1 - s[int(t - 1)]) * b_prime(A_prime)

        # Return RHS value function, period utility
        # function and Lambda function values state by state
        # Xi function is the period utility + Lambda - adjustment cost
        Lambda_B = renter_ind * Lambda_B_rent \
            + (1 - renter_ind) * Lambda_B_norent
        VF_B = renter_ind * VF_B_rent\
            + (1 - renter_ind) * VF_B_norent
        Xi = renter_ind * Xi_rent\
            + (1 - renter_ind) * Xi_norent
        UF_B = renter_ind * UF_B_rent\
            + (1 - renter_ind) * UF_B_norent

        print(np.sum(np.isnan(VF_B)))

        return UC_prime_B, UC_prime_H_B, UC_prime_HFC_B, UC_prime_M_B,\
            Lambda_B, VF_B, Xi, UF_B, zeta

    # @njit
    def UC_cond_DC(UC_prime_B,
                   UC_prime_H_B,
                   UC_prime_HFC_B,
                   UC_prime_M_B,
                   Lambda_B, VF_B,
                   Xi, UF_B):
        """ Indicies of inputs are ordered by:
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

        UC_prime_copi, UC_prime_H_copi, UC_prime_HFC_copi,\
            UC_prime_M_copi, Lambda_B_copi, VF_B_copi, UF_B_copi, Xi_copi,\
            state_index_copi = \
            reshape_out_pi(UC_prime_B),\
            reshape_out_pi(UC_prime_H_B),\
            reshape_out_pi(UC_prime_HFC_B),\
            reshape_out_pi(UC_prime_M_B),\
            reshape_out_pi(Lambda_B), reshape_out_pi(VF_B),\
            reshape_out_pi(UF_B),\
            reshape_out_pi(Xi),\
            reshape_out_pi(X_all_ind[:, 0])

        DB_index_new = np.zeros(len(UC_prime_copi))
        scaling_pi = np.add(sigma_DB_pi * (1 - state_index_copi[:, 0]),
                            sigma_DC_pi * (state_index_copi[:, 0]))
        DB_index_new = state_index_copi[:, 0]
        Xi_copi_temp = np.add(Xi_copi / scaling_pi[:, np.newaxis],
                              - np.max(Xi_copi / scaling_pi[:, np.newaxis],
                                       axis=1)[:, np.newaxis])
        prob_pi = np.exp(Xi_copi_temp)\
            / np.sum(np.exp(Xi_copi_temp), axis=1)[:, np.newaxis]

        UC_prime_cpi = einsum_row(prob_pi, UC_prime_copi)
        UC_prime_H_cpi = einsum_row(prob_pi, UC_prime_H_copi)
        UC_prime_HFC_cpi = einsum_row(prob_pi, UC_prime_HFC_copi)
        UC_prime_M_cpi = einsum_row(prob_pi, UC_prime_M_copi)
        Lambda_B_cpi = einsum_row(prob_pi, Lambda_B_copi)
        VF_B_cpi = einsum_row(prob_pi, VF_B_copi)
        UF_B_cpi = einsum_row(prob_pi, UF_B_copi)
        Xi_cpi = einsum_row(prob_pi, Xi_copi)

        # Reshape to condition out V
        UC_prime_cov, UC_prime_H_cov, UC_prime_HFC_cov,\
            UC_prime_M_cov, Lambda_B_cov, VF_B_cov, UF_B_cov, Xi_cov,\
            DB_index_V = reshape_out_V(UC_prime_cpi),\
            reshape_out_V(UC_prime_H_cpi),\
            reshape_out_V(UC_prime_HFC_cpi),\
            reshape_out_V(UC_prime_M_cpi),\
            reshape_out_V(Lambda_B_cpi),\
            reshape_out_V(VF_B_cpi),\
            reshape_out_V(UF_B_cpi),\
            reshape_out_V(Xi_cpi),\
            reshape_out_V(DB_index_new)

        scaling_v = np.add(sigma_DB_V *
                           np.array((1 -
                                     DB_index_V[:, 0])), sigma_DC_V *
                           np.array((DB_index_V[:, 0])))
        Xi_cov = np.add(Xi_cov / scaling_v[:, np.newaxis],
                        - np.max(Xi_cov / scaling_v[:, np.newaxis],
                                 axis=1)[:, np.newaxis])
        prob_v = np.exp(Xi_cov)\
            / np.sum(np.exp(Xi_cov), axis=1)[:, np.newaxis]

        UC_prime_cv = einsum_row(prob_v, UC_prime_cov)
        UC_prime_H_cv = einsum_row(prob_v, UC_prime_H_cov)
        UC_prime_HFC_cv = einsum_row(prob_v, UC_prime_HFC_cov)
        UC_prime_M_cv = einsum_row(prob_v, UC_prime_M_cov)
        Lambda_B_cv = einsum_row(prob_v, Lambda_B_cov)
        VF_B_cv = einsum_row(prob_v, VF_B_cov)
        UF_B_cv = einsum_row(prob_v, UF_B_cov)
        Xi_cv = einsum_row(prob_v, Xi_cov)

        UC_prime_DCC, UC_prime_H_DCC, UC_prime_HFC_DCC, UC_prime_M_DCC,\
            Lambda_DCC, VF_DCC, UF_DCC\
            = reshape_X_bar(UC_prime_cv),\
            reshape_X_bar(UC_prime_H_cv),\
            reshape_X_bar(UC_prime_HFC_cv),\
            reshape_X_bar(UC_prime_M_cv),\
            reshape_X_bar(Lambda_B_cv),\
            reshape_X_bar(VF_B_cv),\
            reshape_X_bar(UF_B_cv)
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

        return UC_prime_DCC, UC_prime_H_DCC, UC_prime_HFC_DCC, UC_prime_M_DCC,\
            Lambda_DCC, VF_DCC, Xi_cov_out, Xi_copi_out, UF_DCC,Xi

    @njit(parallel=True, nogil=True)
    def UC_cond_all(t, UC_prime_DCC, UC_prime_H_DCC,
                    UC_prime_HFC_DCC, UC_prime_M_DCC,
                    Lambda_DCC, VF_DCC, UF_DCC):
        """Generate RHS T_R Euler equation conditioned on:
                - housing stock taken into time T_R (H_{TR-1})
                - DC assets (before returns) taken into into time T_R
                - T_R-1 housing stock
                - liquid assets taken into time T_R (before returns)
                - T_R -1 wage shock, alpha, beta shock, Pi
                - T_R- 1 house price
                - DB/DC

                First index of output corresponds to discrete index in cart
                prod of disctete exog states

        Parameters
        ----------
        UC_prime_B: 1D array (flat 9D array)
        UC_prime_H_B: 1D array (flat 9D array)
        UC_prime_HFC_B: 1D array (flat 9D array)
        Lambda_B: 1D array (flat 9D array)
        Xi: 1D array (flat 9D array)

        Returns
        -------
        UC_prime_out: 9D array

        UC_prime_eta_out: 9D array

        Lambda: 9D array

        """
        UC_prime = np.empty(len(X_all_hat_vals))
        UC_prime_H = np.empty(len(X_all_hat_vals))
        UC_prime_HFC = np.empty(len(X_all_hat_vals))
        UC_prime_M = np.empty(len(X_all_hat_vals))
        Lambda = np.empty(len(X_all_hat_vals))
        VF = np.empty(len(X_all_hat_vals))
        UF = np.empty(len(X_all_hat_vals))

        for i in prange(len(X_all_hat_vals)):

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
            E_ind = X_all_hat_ind[i][1]
            alpha_ind = X_all_hat_ind[i][2]
            beta_ind = X_all_hat_ind[i][3]
            DB_ind = int(0)
            a_ind = int(X_all_hat_ind[i][5])
            h_ind = int(X_all_hat_ind[i][7])
            point = X_prime_vals[i][:, 0:3]
            r_m_prime = X_prime_vals[i][:, 3]

            E_ind_p, alpha_ind_p, beta_ind_p\
                = E_ind, alpha_ind, beta_ind

            # gen UC_prime unconditioned on T_R -1 income
            U1, U2, U3, U4, L, V, U\
                = eval_linear(X_DCQ_W,
                              UC_prime_DCC[DB_ind,
                                           E_ind_p,
                                           alpha_ind_p,
                                           beta_ind_p,
                                           a_ind, :,
                                           h_ind, :, :],
                              point,
                              xto.LINEAR),\
                eval_linear(X_DCQ_W,
                            UC_prime_H_DCC[DB_ind,
                                           E_ind_p,
                                           alpha_ind_p,
                                           beta_ind_p,
                                           a_ind, :,
                                           h_ind, :, :],
                            point, xto.LINEAR),\
                eval_linear(X_DCQ_W,
                            UC_prime_HFC_DCC[DB_ind,
                                             E_ind_p, alpha_ind_p,
                                             beta_ind_p, a_ind, :,
                                             h_ind, :, :],
                            point, xto.LINEAR),\
                eval_linear(X_DCQ_W,
                            UC_prime_M_DCC[DB_ind,
                                           E_ind_p, alpha_ind_p,
                                           beta_ind_p, a_ind, :,
                                           h_ind, :, :],
                            point, xto.LINEAR) * r_m_prime,\
                eval_linear(X_DCQ_W,
                            Lambda_DCC[DB_ind, E_ind_p,
                                       alpha_ind_p, beta_ind,
                                       a_ind, :, h_ind, :, :],
                            point, xto.LINEAR),\
                eval_linear(X_DCQ_W,
                            VF_DCC[DB_ind, E_ind_p,
                                   alpha_ind_p, beta_ind,
                                   a_ind, :, h_ind, :, :],
                            point, xto.LINEAR),\
                eval_linear(X_DCQ_W,
                            UF_DCC[DB_ind, E_ind_p,
                                   alpha_ind_p, beta_ind,
                                   a_ind, :, h_ind, :, :],
                            point, xto.LINEAR)

            UC_prime[i], UC_prime_H[i], UC_prime_HFC[i],\
                UC_prime_M[i], Lambda[i], VF[i], UF[i] \
                = d0(U1, Q_DC_P), d0(U2, Q_DC_P),\
                d0(U3, Q_DC_P), d0(U4, Q_DC_P),\
                d0(L, Q_DC_P), d0(V, Q_DC_P), d0(U, Q_DC_P)

        #UC_prime[UC_prime<=0] = 1e-250
        #UC_prime_H[UC_prime_H<=0]= 1e-250
        #UC_prime_HFC[UC_prime_HFC<=0] = 1e-250
        #UC_prime_M[UC_prime_M<=0] = 1e-250
        #Lambda[Lambda<=0]= 1e-250

        return UC_prime, UC_prime_H, UC_prime_HFC, UC_prime_M, Lambda, VF, UF

    # Define function to solve worker policies via backward iteration

    def solve_LC_model(load_ret, ret_path):

        if load_ret == 1:

            ret_pols = pickle.load(open("{}/ret_pols.pol".format(ret_path), "rb"))
            (UC_prime, UC_prime_H,UC_prime_HFC, UC_prime_M, Lambda, VF) = ret_pols

        else:
                #a_noadj, c_noadj, etas_noadj, a_adj_uniform, c_adj_uniform,\
                #H_adj_uniform, zeta_nl, c_adj_uniform_nl, H_adj_uniform_nl,\
                #h_prime_rent

            UC_prime, UC_prime_H,UC_prime_HFC, UC_prime_M, Lambda, VF = gen_R_pol()

            ret_pols = (UC_prime, UC_prime_H,UC_prime_HFC, UC_prime_M, Lambda, VF)
            pickle.dump(ret_pols, open("{}/ret_pols.pol".format(ret_path), "wb"))

        UF = VF

        start1 = time.time()
        for Age in np.arange(int(tzero), int(R))[::-1]:

            start = time.time()
            t = Age
            print(Age)

            # Step 1: Evaluate optimal mortgage choice func
            mort_func = eval_pol_mort_W(t, UC_prime, UC_prime_M)

            # Step 2: Evaluate renter policy
            H_rent = eval_rent_pol_W(UC_prime)

            # Step 2: Evaluate policy for non- housing adjusters
            C_noadj, etas_noadj, Aprime_noadj = eval_policy_W_noadj(
                t, mort_func, UC_prime, UC_prime_H, UC_prime_HFC, UC_prime_M, UF)

            # Step 3: Evaluate policy for housing adjusters
            C_adj, H_adj, Aprime_adj = eval_policy_W_adj(t,
                                                         mort_func,
                                                         UC_prime,
                                                         UC_prime_H,
                                                         UC_prime_HFC,
                                                         UC_prime_M)

            print("Solved eval_policy_W of age {} in {} seconds".
                  format(Age, time.time() - start))

            noadj_pols = np.stack([reshape_nadj_RHS(C_noadj),
                                   reshape_nadj_RHS(etas_noadj),
                                   reshape_nadj_RHS(Aprime_noadj)], axis=3)
            adj_pols = np.stack([reshape_adj_RHS(C_adj),
                                 reshape_adj_RHS(H_adj),
                                 reshape_adj_RHS(Aprime_adj)], axis=3)
            rent_pols = reshape_rent_RHS(H_rent)

            # value funcs are indexed by
            # DB(0)xE(1)xA(2)xB(3)xPi(4)xA(5)xA_DC(6)xH(7)xQ(8)xM(9)
            # re-order to
            # DB(0)xE(1)xA(2)xB(3)xPi(4)xQ(8)xA(5)xA_DC(6)xH(7)xM(9)

            val_funcs = np.stack([VF.reshape(all_state_shape)
                                  .transpose((0, 1, 2, 3, 4, 8, 5, 6, 7, 9))
                                  .reshape((v_func_shape)),
                                  Lambda
                                  .reshape(all_state_shape)
                                  .transpose((0, 1, 2, 3, 4, 8, 5, 6, 7, 9))
                                  .reshape((v_func_shape))], axis=5)

            # Step 4: Evaulate unconditioned RHS of Euler equations
            start = time.time()
            UC_prime_B, UC_prime_H_B,\
                UC_prime_HFC_B, UC_prime_M_B, Lambda_B, VF_B, Xi, UF_B, zeta\
                = gen_RHS_UC_func(t,
                                  noadj_pols,
                                  adj_pols,
                                  rent_pols,
                                  val_funcs,
                                  points_noadj_vec[int(t - tzero)],
                                  points_adj_vec[int(t - tzero)],
                                  points_rent_vec[int(t - tzero)],
                                  A_prime[int(t - tzero)])

            print("Solved gen_RHS_UC_func of age {} in {} seconds".
                  format(Age, time.time() - start))

            #print(np.sum(np.isnan(UC_prime_B)))
            # Step 5: Condition out discrete choice and preference shocks
            start = time.time()
            UC_prime_DCC, UC_prime_H_DCC, UC_prime_HFC_DCC,\
                UC_prime_M_DCC, Lambda_DCC, VF_DCC,\
                Xi_cov, Xi_copi, UF_DCC,Xi_copi_temp = UC_cond_DC(UC_prime_B,
                                                     UC_prime_H_B,
                                                     UC_prime_HFC_B,
                                                     UC_prime_M_B,
                                                     Lambda_B, VF_B, Xi, UF_B)
            
            print("Solved UC_cond_DC of age {} in {} seconds".
                  format(Age, time.time() - start))
            #print(np.sum(np.isnan(UC_prime_DCC)))

            # Step 6: Condition out DC return and house price shocks
            start = time.time()
            UC_prime, UC_prime_H, UC_prime_HFC, UC_prime_M, Lambda, VF, UF\
                = UC_cond_all(t,
                              UC_prime_DCC,
                              UC_prime_H_DCC,
                              UC_prime_HFC_DCC,
                              UC_prime_M_DCC,
                              Lambda_DCC, VF_DCC, UF_DCC)
            
            print("Solved UC_cond_all of age {} in {} seconds".
                  format(Age, time.time() - start))
            #print(np.sum(np.isnan(UC_prime)))
            print(time.time() - start)

            # Step 7: Save policy functions to array
            policy_Aprime_noadj[int(t - tzero), :] = Aprime_noadj\
                .astype(np.float32)
            policy_C_noadj[int(t - tzero), :] = C_noadj\
                .astype(np.float32)
            policy_H_adj[int(t - tzero), :] = H_adj.astype(np.float32)
            policy_Aprime_adj[int(t - tzero), :] = Aprime_adj\
                .astype(np.float32)
            policy_C_adj[int(t - tzero), :] = C_adj\
                .astype(np.float32)
            policy_Xi_cov[int(t - tzero), :] = Xi_cov\
                .astype(np.float32)
            policy_Xi_copi[int(t - tzero), :] = Xi_copi\
                .astype(np.float32)
            policy_etas_noadj[int(t - tzero), :] = etas_noadj\
                .astype(np.float32)
            policy_zeta[int(t - tzero), :] = zeta.reshape(all_state_shape_hat).astype(np.float32)
            policy_H_rent[int(t - tzero), :] = H_rent.astype(np.float32)

            if t == tzero:
                policy_VF[:] = VF.reshape(
                    (len(DB),
                     grid_size_W,
                     grid_size_alpha,
                     grid_size_beta,
                     len(Pi),
                        grid_size_A,
                        grid_size_DC,
                        grid_size_H,
                        grid_size_Q,
                        grid_size_M))
        print("Solved lifecycle model in {} seconds"
              .format(time.time() - start1))

        return [policy_C_noadj, policy_etas_noadj, policy_Aprime_noadj,
                policy_C_adj, policy_H_adj, policy_Aprime_adj,
                policy_H_rent, policy_zeta,
                policy_Xi_cov, policy_Xi_copi, policy_VF]

    return solve_LC_model


def generate_worker_pols(og, load_retiree=1, ret_sol_path = '/scratch/pv33'):

    gen_R_pol = retiree_func_factory(og)
    solve_LC_model = worker_solver_factory(og, gen_R_pol)
    policies = solve_LC_model(load_retiree,ret_sol_path)

    return policies
