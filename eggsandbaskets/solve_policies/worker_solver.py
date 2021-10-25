""" Module generates operator to solve worker policy functions

Functions: worker_solver_factory
	Generates  operator solve an instance of LifeCycleModel

Example use:

gen_R_pol = retiree_func_factory(lifecycle_model)

solve_LC_model = worker_solver_factory(lifecycle_model,gen_R_pol)

solve_LC_model = worker_solver_factory(og, world_comm, comm, gen_R_pol,
					scr_path = scr_path,
					gen_newpoints = gen_newpoints,
					verbose = False)
"""

# Import packages
from mpi4py import MPI
from pathlib import Path
import dill as pickle
from interpolation import interp
from interpolation.splines import extrap_options as xto
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from quantecon.optimize.root_finding import brentq
from numba import njit, prange
import time
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

from util.helper_funcs import gen_policyout_arrays, d0, interp_as,\
    gen_reshape_funcs, _my_einsum_row, eval_linear_c
from solve_policies.retiree_solver import retiree_func_factory


def worker_solver_factory(og,
                          world_comm,
                          comm,
                          gen_R_pol,
                          jobfs_path,
                          verbose=False, 
                          plot_vf = False):
    """Generates operator that solves  worker policies

    Parameters
    ----------
    og : LifeCycleModel
              Instance of LifeCycleModel
    world_comm: MPI Communicator
		Communicator for *all* CPUs across SMM estimation
    comm: MPI Communicator
                    Layer 2 communicator class
                    Communicator to solve this instance of LifeCycleModel
    gen_R_pol: Function
         function generates retiring (T_R) age policy and RHS Euler
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
    mod_name = og.mod_name
    acc_ind = og.acc_ind
    ID = og.ID

    # Define and create path where policies are saved
    pol_path_id = jobfs_path + '/' + mod_name + '/' + ID\
        + '_acc_' + str(acc_ind[0])
    Path(pol_path_id).mkdir(parents=True, exist_ok=True)

    # Functions
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

    # Parameters
    s = og.parameters.s
    delta_housing, tau_housing, def_pi = og.parameters.delta_housing,\
        og.parameters.tau_housing,\
        og.parameters.def_pi
    beta_bar, alpha_bar = og.parameters.beta_bar, og.parameters.alpha_bar

    # Grid size parameters
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

    # Exogenous shocks
    beta_hat, P_beta, beta_stat = og.st_grid.beta_hat,\
        og.st_grid.P_beta,\
        og.st_grid.beta_stat
    alpha_hat, P_alpha, alpha_stat = og.st_grid.alpha_hat,\
        og.st_grid.P_alpha,\
        og.st_grid.alpha_stat
    X_r, P_r = og.st_grid.X_r, og.st_grid.P_r
    Q_shocks_r, Q_shocks_P = og.st_grid.Q_shocks_r,\
        og.st_grid.Q_shocks_P
    Q_DC_shocks, Q_DC_P = og.cart_grids.Q_DC_shocks,\
        og.cart_grids.Q_DC_P
    r_m_prime = og.cart_grids.r_m_prime
    EBA_P, EBA_P2 = og.cart_grids.EBA_P, og.cart_grids.EBA_P2
    E, P_E, P_stat = og.st_grid.E, og.st_grid.P_E, og.st_grid.P_stat

    # Small grids
    A_DC = og.grid1d.A_DC
    H = og.grid1d.H
    A = og.grid1d.A
    M = og.grid1d.M
    Q = og.grid1d.Q
    H_R = og.grid1d.H_R
    W_W = og.grid1d.W_W
    V, Pi, DB = og.grid1d.V, og.grid1d.Pi, og.grid1d.DB

    # Medium and interpolation grids
    X_cont_W = og.interp_grid.X_cont_W
    X_cont_W_bar = og.interp_grid.X_cont_W_bar
    X_cont_W_hat = og.interp_grid.X_cont_W_hat
    X_DCQ_W = og.interp_grid.X_DCQ_W
    X_QH_W = og.interp_grid.X_QH_W
    X_cont_WAM = og.interp_grid.X_cont_WAM
    X_cont_WAM2 = og.interp_grid.X_cont_WAM2
    X_cont_AH = og.interp_grid.X_cont_AH
    X_W_contgp = og.interp_grid.X_W_contgp

    # Large grids
    if comm.rank == 1:
        X_all_C_ind = og.BigAssGrids.X_all_C_ind_f()\
            .astype(np.int32)
    if comm.rank == 0:
        X_all_hat_ind = og.BigAssGrids.X_all_hat_ind_f()\
            .astype(np.int32)
        x_all_hat_ind_len = len(X_all_hat_ind)
        X_all_B_ind = og.BigAssGrids.X_all_B_ind_f().astype(np.int32)

        X_W_bar_hdjex_ind = og.BigAssGrids.X_W_bar_hdjex_ind_f()\
            .astype(np.int32)
        x_all_ind_len = og.BigAssGrids.X_all_ind_f(ret_len=True)
        X_all_ind = og.BigAssGrids.X_all_ind_f(ret_len=False)
        len_X_W_bar_hdjex_ind = len(X_W_bar_hdjex_ind)
        X_all_ind_is = np.arange(x_all_ind_len)
        X_all_ind_split = np.array_split(X_all_ind, comm.size, axis=0)

        # Large grids that are scattered to layer 2 workers
        X_W_bar_hdjex_ind_split = np.array_split(X_W_bar_hdjex_ind, comm.size,
                                                 axis=0)
        X_all_hat_ind_split = np.array_split(X_all_hat_ind, comm.size,
                                             axis=0)

    else:
        X_W_bar_hdjex_ind = None
        X_W_bar_hdjex_ind_split = None
        len_X_W_bar_hdjex_ind = None
        x_all_hat_ind_len = None
        x_all_ind_len = None
        X_all_hat_ind = None
        X_all_ind_split = None
        X_all_ind = None
        X_all_ind_is = None
        X_all_hat_ind_split = None

    # Scatter grids to workers, broadcast constants
    my_X_W_bar_hdjex_ind = comm.scatter(X_W_bar_hdjex_ind_split, root=0)
    my_X_all_ind = comm.scatter(X_all_ind_split, root=0)
    my_X_all_hat_ind = comm.scatter(X_all_hat_ind_split, root=0)

    len_X_W_bar_hdjex_ind = comm.bcast(len_X_W_bar_hdjex_ind, root=0)
    x_all_hat_ind_len = comm.bcast(x_all_hat_ind_len, root=0)
    x_all_ind_len = comm.bcast(x_all_ind_len, root=0)

    my_X_all_ind = my_X_all_ind.astype(np.int32)
    my_X_all_hat_ind = my_X_all_hat_ind.astype(np.int32)

    # Load reset array shapes
    all_state_shape, all_state_shape_hat, v_func_shape,\
        all_state_A_last_shape, policy_shape_nadj,\
        policy_shape_adj, policy_shape_rent, prob_v_shape, prob_pi_shape\
        = gen_policyout_arrays(og)

    # Load re-shaping functions
    reshape_out_pi, reshape_out_V, reshape_X_bar,\
        reshape_make_Apfunc_last, reshape_make_h_last,\
        reshape_vfunc_points, reshape_nadj_RHS, reshape_rent_RHS,\
        reshape_adj_RHS, reshape_RHS_Vfunc_rev, reshape_RHS_UFB,\
        reshape_vfunc_points, _rehape_adj_post_interp = gen_reshape_funcs(og)

    del og, X_all_hat_ind_split, X_W_bar_hdjex_ind,\
        X_all_hat_ind, X_all_ind, X_all_ind_split

    gc.collect()

    # Generic functions

    @njit
    def _noadj_pre_interp_reshape(policy, x_all_hat_ind_len):

        policy_reshaped = policy.reshape(all_state_shape)
        policy_reshaped = np.transpose(policy_reshaped,
                                       (0, 1, 2, 3, 4, 6, 7, 8, 9, 5))
        policy_reshaped_1 = np.copy(policy_reshaped)\
            .reshape((int(x_all_hat_ind_len
                          / grid_size_A), int(grid_size_A)))

        return policy_reshaped_1

    @njit
    def gen_wage_vec(j, my_X_all_ind):

        wage_vector = yvec(int(j), E[my_X_all_ind[:, 1]])

        return wage_vector

    @njit
    def my_gen_points_for_age(j, my_X_all_ind):
        """
        Generates 'wealth' points for each j (age) over which
        the marginal utilities and value functions are evaluated

        Function takes in my worker part of the full grid of points
        and returns points for the worker

        Parameters
        ----------
        j: float
         age of worker (real age -1 as first year is 0)

        Returns
        ----------
        my_points_noadj_vec:  flat 12D array
        my_points_adj_vec:    flat 12D array
        my_A_prime:            flat 12D array

        Each element of 11D X_all_ind[i] is a
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
        10 - M

        Notes
        -----
        Points are defined on states at beggining of time t, but after
        vol. cont and risk share decision has been made for time t. Thus
        the risk share decision affects the stochastic returns at time t+1
        to the t+1 pension balance.

        Note by convention, we assume the state of housing at beggining
        f time t is always before depreciation on the fundamental grids. Thus
        housing stock that goes into evaluating total wealth is adjusted
        for dep.

        Mortgage leverage here is after interest and house repreciation

        """

        # Make vectors of compulsory and contribution rates and
        # adjustment cost, wage
        v_Sv = np.full(len(my_X_all_ind), v_S)
        v_Ev = np.full(len(my_X_all_ind), v_E)
        v = V[my_X_all_ind[:, 4]]
        tau_housing_vec = np.full(len(my_X_all_ind), tau_housing)
        wage = gen_wage_vec(j, my_X_all_ind)

        # Make vector of 1- total contribution rate (as % of wage )
        after_total_cont_rate = np.ones(len(my_X_all_ind)) - v - v_Sv - v_Ev

        # Pull out vectors of time t
        # account type, mortgage balance leverage, housing stock
        # after depreciation, house price, total mortgage liability,
        # DC balance
        DC_ind = my_X_all_ind[:, 0]
        m = M[my_X_all_ind[:, 10]]
        h = H[my_X_all_ind[:, 8]] * (1 - delta_housing)
        q = Q[my_X_all_ind[:, 9]]
        m_val = h * m * q
        a_dc = A_DC[my_X_all_ind[:, 7]]

        # Generate empty vetors to fill with 'wealth' points
        my_points_noadj_vec, my_points_adj_vec\
            = np.empty((len(my_X_all_ind), 2)),\
            np.empty((len(my_X_all_ind), 2))

        # Total liquid wealth (cash in hand) for non-adjuster
        # and adjuster
        my_points_noadj_vec[:, 0] = A[my_X_all_ind[:, 6]] * (1 + r) \
            + after_total_cont_rate * wage

        my_points_noadj_vec[:, 0][my_points_noadj_vec[:, 0] <= 0] = A_min

        my_points_adj_vec[:, 0] = my_points_noadj_vec[:, 0] + q * h * (1 - m)

        # Next period DC assets (before returns)
        # (recall domain of policy functions from def of eval_policy_W)
        my_points_noadj_vec[:, 1] = a_dc + v * \
            wage + (v_S + v_E) * wage * DC_ind
        my_points_adj_vec[:, 1] = a_dc + v * wage + (v_S + v_E) * wage * DC_ind

        # Total bequest if agent dies before beggining the period
        # (but after returns have been accumulated)
        my_A_prime = my_points_adj_vec[:, 0] 

        # Ravel all the points so easier to Gatherv via mpi
        return np.ravel(my_points_noadj_vec),\
            np.ravel(my_points_adj_vec), my_A_prime

    def gen_points_for_age(j, comm):
        """ Function evalutes wealth points at full state for each age
                across workers

        Parameters
        ----------
        j: float
         age
        comm: Layer 2 communicator


        Returns
        -------
        On layer 2 root node:
         points_noadj_vec:  flat 12D array
         points_adj_vec:    flat 12D array
         A_prime:            flat 12D array
        On layer 2 workers:
         None

        """

        points_noadj_vec, points_adj_vec = np.empty(int(x_all_ind_len * 2)),\
            np.empty(int(x_all_ind_len * 2))
        A_prime = np.empty(int(x_all_ind_len))

        # Generate my points on each worker
        my_points_noadj_vec, my_points_adj_vec, my_A_prime\
            = my_gen_points_for_age(j, my_X_all_ind)

        # Gen sound counts
        sendcounts_1 = np.array(comm.gather(len(my_points_noadj_vec), 0))
        sendcounts_2 = np.array(comm.gather(len(my_A_prime), 0))

        # Gather from each worker at root
        comm.Gatherv(np.ascontiguousarray(my_points_noadj_vec),
                     recvbuf=(points_noadj_vec, sendcounts_1), root=0)
        comm.Gatherv(np.ascontiguousarray(my_points_adj_vec),
                     recvbuf=(points_adj_vec, sendcounts_1), root=0)
        comm.Gatherv(np.ascontiguousarray(my_A_prime),
                     recvbuf=(A_prime, sendcounts_2), root=0)

        del my_points_noadj_vec, my_points_adj_vec, my_A_prime
        gc.collect()

        # Reshape points to wide on root and return
        if comm.rank == 0:

            points_adj_vec = points_adj_vec\
                .reshape((1, grid_size_W,
                          grid_size_alpha,
                          grid_size_beta,
                          len(V),
                          len(Pi),
                          grid_size_A,
                          grid_size_DC,
                          grid_size_H,
                          grid_size_Q,
                          grid_size_M, 2))
            # Change adjuster coordinates to following order
            #  DB(0),E(1), Alpha(2), Beta(3), Pi(5), Q(9),\
            #  M(10), V(4), A(6), A_DC(7), H(8), 2D points(11)

            points_adj_vec = points_adj_vec\
                .transpose((0, 1, 2, 3, 5, 9, 10, 4, 6, 7, 8, 11))

            # Mow reshape to:
            # |DBx Ex Alpha x BetaxPi xQxM| x|VxAxA_DCxH|x2
            # Recall each i \in |DBx Ex Alpha x BetaxPi xQxM|
            # adjuster function will be reshaped to a
            # function on Wealth x A_DC
            points_adj_vec = points_adj_vec\
                .reshape((int(1 * grid_size_W * grid_size_alpha *
                              grid_size_beta * len(Pi) *
                              grid_size_Q * grid_size_M),
                          int(grid_size_H * len(V) *
                              grid_size_A *
                              grid_size_DC), 2))
            # Reshape the renter points
            points_rent_vec = np.copy(points_adj_vec).reshape((1, grid_size_W,
                                                               grid_size_alpha,
                                                               grid_size_beta,
                                                               len(V),
                                                               len(Pi),
                                                               grid_size_A,
                                                               grid_size_DC,
                                                               grid_size_H,
                                                               grid_size_Q,
                                                               grid_size_M, 2))
            # Change renter coordinates to:
            #  DB(0), E(1), Alpha(2), Beta(3), Pi(5), Q(9),\
            #  V(4), A(6), A_DC(7), H(8), M(10), points(2)
            points_rent_vec = points_rent_vec\
                .transpose((0, 1, 2, 3, 5, 9, 4, 6, 7, 8, 10, 11))

            points_rent_vec = points_rent_vec\
                .reshape((int(1 * grid_size_W * grid_size_alpha *
                              grid_size_beta * len(Pi) *
                              grid_size_Q),
                          int(grid_size_H * len(V) *
                              grid_size_A * grid_size_M *
                              grid_size_DC), 2))

            # Reshape the no adjustment points
            points_noadj_vec = points_noadj_vec\
                .reshape((1, grid_size_W,
                          grid_size_alpha,
                          grid_size_beta,
                          len(V),
                          len(Pi),
                          grid_size_A,
                          grid_size_DC,
                          grid_size_H,
                          grid_size_Q,
                          grid_size_M, 2))

            # Change noadj coordinates to:
            # DB(0), E(1), Alpha(2), Beta(3), Pi(5), H(8), Q(9),M(10)\
            #  V(4), A(6), A_DC(7), points(11)
            points_noadj_vec = points_noadj_vec\
                .transpose((0, 1, 2, 3, 5, 8, 9, 10, 4, 6, 7, 11))
            points_noadj_vec = points_noadj_vec\
                .reshape((int(1 * grid_size_W * grid_size_alpha *
                              grid_size_beta * len(Pi) *
                              grid_size_H * grid_size_Q
                              * grid_size_M),
                          int(len(V) *
                              grid_size_A *
                              grid_size_DC), 2))

            return points_noadj_vec, points_adj_vec, points_rent_vec, A_prime
        else:
            return None, None, None, None

    @njit
    def gen_x_iid_prime_vals(pi_ind, ADC_ind, q_ind, m_ind):
        """
        Lazy evalutes t+1 states that are subject to iid shocks (DC balance,
        house price leverage and rate of mortgage interest)
        after shock has been realised. The shocks are:
                - house price shock
                - low and high risk DC return shock
                - mortgage return shock
        ""

        pi_ind: int
         Index of Pi (t-1)
        ADC_ind: int
         Index of A_DC (t-1, before returns) taken into t
        q_ind: int
         Index of Q (t-1)
        m_ind: int
         Index of M (t-1) end of period in t-1 prices

        Returns
        -------
        TBC

        Notes
        -----

        M is end of t period mortgage leverage but at time t period prices.
        M_prime is beggining of t+1 period leverage in t+1 house prices.

        Note t+1 beggining of period leverage is defined in terms of dep.
        housing stock.

        """
        r_share = Pi[pi_ind]
        ADC_in = A_DC[ADC_ind]
        q_in = Q[q_ind]
        m_in = M[m_ind]

        q_t_arr = np.full(len(Q_DC_shocks[:, 2]), q_in)
        r_H_arr = np.full(len(Q_DC_shocks[:, 2]), r_H)
        Q_prime = (1 + r_H_arr + Q_DC_shocks[:, 2]) * q_t_arr

        risky_share_arr = np.full(len(Q_DC_shocks[:, 2]), r_share)

        A_DC_returns = (1 - risky_share_arr) * Q_DC_shocks[:, 0] +\
            risky_share_arr * Q_DC_shocks[:, 1]

        A_DC_prime = A_DC_returns * np.full(len(Q_DC_shocks[:, 2]), ADC_in)
        M_prime = r_m_prime * m_in * q_t_arr / \
            ((1 - delta_housing) * Q_prime)

        return np.column_stack((A_DC_prime, Q_prime, M_prime))

    @njit
    def uc_inv(uc, s, alpha):
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

        args = (s, alpha, uc)
        if ces_c1(C_min, s, alpha, uc) < 0:
            return C_min
        elif ces_c1(C_max, s, alpha, uc) > 0:
            return C_max
        elif ces_c1(C_min, s, alpha, uc) * ces_c1(C_max, s, alpha, uc) < 0:
            return brentq(ces_c1, C_min, C_max, args=args, disp=False)[0]
        else:
            return C_min

    # First order conditions

    @njit
    def FOC_mortgage(m, UC_prime_func1D, UC_prime_M_func1D):
        """Error of first order cond. of mortgage interior solution (equation x
        in paper)

        Parameters
        ----------
        m : float64
                        Mortgage leverage taken into t+1, at t prices before t+1 dep.

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

        error = np.interp(m, M, UC_prime_func1D)\
            - np.interp(m, M, UC_prime_M_func1D)
        return error

    @njit
    def FOC_renters(cons, h, q, alpha_housing, UC_prime_func1D):
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
    def FOC_housing_adj(x_prime,
                        a_dc,
                        h,
                        q,
                        alpha_housing,
                        beta,
                        mort_func1D,
                        UC_prime_func2D,
                        UC_prime_H_func2D,
                        UC_prime_M_func2D,
                        t,
                        ret_cons=False):
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

        """

        # Maximum leverage at constraint
        max_loan = (1 - phi_c)

        # Evaluate constrained leverage
        m_prime_adj_tilde = eval_linear_c(X_cont_AH,
                                          mort_func1D,
                                          np.array([x_prime, h]),
                                          xto.LINEAR)
        m_prime_tilde = max(0, min(m_prime_adj_tilde, max_loan))

        # Evaluate RHS (t-1) marginal util of consumption
        uc_prime_adj = max(1e-250, beta * eval_linear_c(X_cont_WAM2,
                                                        UC_prime_func2D,
                                                        np.array([x_prime, h, m_prime_tilde]),
                                                        xto.LINEAR))
        # Evaluate consumption
        c_t = max(C_min, uc_inv(uc_prime_adj, h, alpha_housing))

        # Mutlipler active when mortage
        # constrained by collat. constraint
        ref_const_FOC = 0

        if m_prime_tilde >= (1 - phi_c):
            uc_prime_m_maxloan = max(1e-250,
                                     beta * eval_linear_c(X_cont_WAM2,
                                                          UC_prime_M_func2D,
                                                          np.array([x_prime,
                                                                    h,
                                                                    max_loan]),
                                                          xto.LINEAR))  # check this

            ref_const_FOC = q * (1 - phi_c)\
                * max(0,(uc_prime_adj - uc_prime_m_maxloan))

        # Evaluate of RHS (t+1) of housing marginal shadow value
        UC_prime_H_RHS = beta * eval_linear_c(X_cont_WAM2, UC_prime_H_func2D, np.array([
                                              x_prime, h, m_prime_tilde]), xto.LINEAR)

        marginal_cost_housing = uc_prime_adj * q * (1 + tau_housing)
        marginal_return_housing = UC_prime_H_RHS + uh(c_t, h, alpha_housing) + ref_const_FOC

        if ret_cons:
            return c_t
        else:
            return marginal_cost_housing - marginal_return_housing

    @njit
    def FOC_housing_adj_bind(c,
                             x_prime,
                             a_dc,
                             h,
                             q,
                             alpha_housing,
                             beta,
                             mort_func1D,
                             UC_prime_func2D,
                             UC_prime_H_func2D,
                             UC_prime_M_func1D,
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

        See note for `FOC_housing_adj'
        """

        # Binding liquid asset FOC binding implies
        # binding mortgage FOC, first calculate binding point

        max_loan = (1 - phi_c)

        uc_prime = uc(c, h, alpha_housing)

        m_prime_tilde = min(max(0,np.interp(uc_prime, UC_prime_M_func1D, M)),max_loan)



        # Calcuate RHS of housing Euler at m_prime

        UC_prime_H_RHS = beta * eval_linear_c(X_cont_WAM2, UC_prime_H_func2D,
                                              np.array([x_prime, h, m_prime_tilde]),
                                              xto.LINEAR)


        # Marginal value of relaxing collat constraint
        ref_const_FOC = 0

        if m_prime_tilde >= (1 - phi_c):
            ref_const_FOC = q * (1 - phi_c)\
                * (max(0, uc_prime - np.interp(m_prime_tilde,M, UC_prime_M_func1D)))

        marginal_cost_housing = uc_prime * q * (1 + tau_housing)
        marginal_return_housing = UC_prime_H_RHS + uh(c, h, alpha_housing)\
            + ref_const_FOC

        return marginal_cost_housing - marginal_return_housing

    # Define functions that solve the policy functions and inverse functions

    @njit
    def eval_M_prime_findroot(UC_prime_func1D, UC_prime_M_func1D):
        """Solve mortgage FOC for unconstrained mortgage solution."""

        UC_prime_m_max = UC_prime_func1D[-1]
        UC_prime_M_m_max = UC_prime_M_func1D[-1]
        UC_prime_m_min = UC_prime_func1D[0]
        UC_prime_M_m_min = UC_prime_M_func1D[0]

        if FOC_mortgage(0, UC_prime_func1D, UC_prime_M_func1D)\
                * FOC_mortgage(M[-1], UC_prime_func1D, UC_prime_M_func1D) < 0:
            return brentq(FOC_mortgage, 0, M[-1],
                          args=(UC_prime_func1D,
                                UC_prime_M_func1D), disp=False)[0]
        elif UC_prime_m_max >= UC_prime_M_m_max:
            return M[-1]
        else:
            return 0


    @njit
    def eval_M_prime(t, UC_prime, UC_prime_M):
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
            mort_prime[i] = eval_M_prime_findroot(UC_prime_func1D,
                                                  UC_prime_M_func1D)

            if mort_prime[i] == np.nan:
                mort_prime[i] = 0

        mort_tilde_func = mort_prime.reshape((len(DB), grid_size_W,
                                              grid_size_alpha,
                                              grid_size_beta,
                                              len(Pi),
                                              grid_size_A,
                                              grid_size_DC,
                                              grid_size_H,
                                              grid_size_Q))
        return mort_tilde_func

    @njit
    def eval_rent_pol_W(UC_prime, VF_prime):
        """Creates renter policy function as function of wealth_t, q_t, dc_t+1
        and exogenous states at t using endog grid method."""

        a_end_1 = np.zeros(len(X_all_B_ind))  # Endog grid for a_t
        v_end_1 = np.zeros(len(X_all_B_ind))
        uc_end_1 = np.zeros(len(X_all_B_ind))
        UC_prime_func_ex = UC_prime.reshape(all_state_shape)
        VF_prime_func_ex = VF_prime.reshape(all_state_shape)

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
            VF_prime_func1D = VF_prime_func_ex[DB_ind,
                                               E_ind, alpha_ind,
                                               beta_ind, Pi_ind, :,
                                               adc_ind, 0, q_ind, 0]

            # calculate optimal consumption for h and q value
            c_t = max(C_min, ch_ser(h, alpha_housing, phi_r * q))

            # calculate optimal a_t+1
            a_prime_1 = FOC_renters(c_t, h, q, alpha_housing,
                                    UC_prime_func1D)
            a_prime = max(min(a_prime_1, A_max_W), A_min)

            # generate endogenous grid
            a_end_1[i] = c_t + a_prime + q * phi_r * h
            v_end_1[i] = u(c_t, h, alpha_housing) + \
                beta * interp_as(A, VF_prime_func1D, np.array([a_prime]))[0]
            uc_end_1[i] = uc(c_t, h, alpha_housing)

        a_end_2 = np.copy(a_end_1).reshape((len(DB), grid_size_W,
                                            grid_size_alpha,
                                            grid_size_beta,
                                            len(Pi),
                                            grid_size_DC,
                                            int(grid_size_HS),
                                            grid_size_Q))
        v_end_2 = np.copy(v_end_1).reshape((len(DB), grid_size_W,
                                            grid_size_alpha,
                                            grid_size_beta,
                                            len(Pi),
                                            grid_size_DC,
                                            int(grid_size_HS),
                                            grid_size_Q))
        uc_end_2 = np.copy(uc_end_1).reshape((len(DB), grid_size_W,
                                              grid_size_alpha,
                                              grid_size_beta,
                                              len(Pi),
                                              grid_size_DC,
                                              int(grid_size_HS),
                                              grid_size_Q))

        # make H (housing services) the last ind of the endog asst grid
        a_end_3 = a_end_2.transpose((0, 1, 2, 3, 4, 5, 7, 6))
        v_end_3 = v_end_2.transpose((0, 1, 2, 3, 4, 5, 7, 6))
        uc_end_3 = uc_end_2.transpose((0, 1, 2, 3, 4, 5, 7, 6))

        # reshape so housing service vals are rows
        a_end_4 = np.copy(a_end_3).reshape((int(1 *
                                                grid_size_W *
                                                grid_size_alpha *
                                                grid_size_beta *
                                                len(Pi) *
                                                grid_size_DC *
                                                grid_size_Q),
                                            int(grid_size_HS)))
        v_end_4 = np.copy(v_end_3).reshape((int(1 *
                                                grid_size_W *
                                                grid_size_alpha *
                                                grid_size_beta *
                                                len(Pi) *
                                                grid_size_DC *
                                                grid_size_Q),
                                            int(grid_size_HS)))

        uc_end_4 = np.copy(uc_end_3).reshape((int(1 *
                                                  grid_size_W *
                                                  grid_size_alpha *
                                                  grid_size_beta *
                                                  len(Pi) *
                                                  grid_size_DC *
                                                  grid_size_Q),
                                              int(grid_size_HS)))
        # Empty grid for housing services policy function with
        # a_t indices as rows
        h_prime_func_1 = np.zeros((int(len(DB) * grid_size_W *
                                       grid_size_alpha *
                                       grid_size_beta *
                                       len(Pi) *
                                       grid_size_DC *
                                       grid_size_Q), len(A)))

        v_prime_func_1 = np.zeros((int(len(DB) * grid_size_W *
                                       grid_size_alpha *
                                       grid_size_beta *
                                       len(Pi) *
                                       grid_size_DC *
                                       grid_size_Q), len(A)))

        uc_prime_func_1 = np.zeros((int(len(DB) * grid_size_W *
                                        grid_size_alpha *
                                        grid_size_beta *
                                        len(Pi) *
                                        grid_size_DC *
                                        grid_size_Q), len(A)))

        # Interpolate optimal housing services on endogenous asset grid

        for i in prange(len(h_prime_func_1)):

            a_end_clean = np.sort(a_end_4[i][a_end_4[i] > 0])
            v_end_clean = np.take(v_end_4[i][a_end_4[i] > 0],
                                  np.argsort(a_end_4[i][a_end_4[i] > 0]))
            uc_end_clean = np.take(uc_end_4[i][a_end_4[i] > 0],
                                   np.argsort(a_end_4[i][a_end_4[i] > 0]))

            hs_sorted = np.take(H_R[a_end_4[i] > 0],
                                np.argsort(a_end_4[i][a_end_4[i] > 0]))

            if len(a_end_clean) == 0:
                a_end_clean = np.zeros(1)
                v_end_clean = np.zeros(1)
                uc_end_clean = np.ones(1)
                a_end_clean[0] = A_min
                hs_sorted = np.zeros(1)
                hs_sorted[0] = H_min

            h_prime_func_1[i, :] = interp_as(a_end_clean, hs_sorted, W_W, extrap= True)
            h_prime_func_1[i, :][h_prime_func_1[i, :] <= 0] = H_min

            v_prime_func_1[i, :] = interp_as(a_end_clean, v_end_clean, W_W)

            uc_prime_func_1[i, :] = interp_as(a_end_clean, uc_end_clean, W_W)
            uc_prime_func_1[i, :][uc_prime_func_1[i, :] <= 0] = 1e-200

        h_prime_func_2 = h_prime_func_1.reshape((len(DB),
                                                 grid_size_W,
                                                 grid_size_alpha,
                                                 grid_size_beta,
                                                 len(Pi),
                                                 grid_size_DC,
                                                 grid_size_Q,
                                                 grid_size_A))

        v_prime_func_2 = v_prime_func_1.reshape((len(DB),
                                                 grid_size_W,
                                                 grid_size_alpha,
                                                 grid_size_beta,
                                                 len(Pi),
                                                 grid_size_DC,
                                                 grid_size_Q,
                                                 grid_size_A))
        uc_prime_func_2 = uc_prime_func_1.reshape((len(DB),
                                                   grid_size_W,
                                                   grid_size_alpha,
                                                   grid_size_beta,
                                                   len(Pi),
                                                   grid_size_DC,
                                                   grid_size_Q,
                                                   grid_size_A))

        h_prime_func_3 = h_prime_func_2.transpose((0, 1, 2, 3, 4, 7, 5, 6))
        v_prime_func_3 = v_prime_func_2.transpose((0, 1, 2, 3, 4, 7, 5, 6))
        uc_prime_func_3 = uc_prime_func_2.transpose((0, 1, 2, 3, 4, 7, 5, 6))

        return h_prime_func_3, v_prime_func_3, uc_prime_func_3

    @njit
    def my_invert_noadj_prime_pol(assets_reshaped_1,
                                  vf_reshaped_1,
                                  cons_reshaped_1,
                                  uhdb_reshaped_1,
                                  uc_reshaped_1):
        """Interpolates no adjust policy functions by inverting inverse policy
                (this is the endogenous grid method step)."""

        # generate empty grids to fill with interpolated functions
        Aprime_noadjust_1 = np.empty(
            (len(assets_reshaped_1), int(grid_size_A)))
        V_reshaped_1 = np.empty((len(assets_reshaped_1), int(grid_size_A)))
        C_noadj_1 = np.empty((len(assets_reshaped_1), int(grid_size_A)))
        UHdb_noadj_1 = np.empty((len(assets_reshaped_1), int(grid_size_A)))
        UC_noadj_1 = np.empty((len(assets_reshaped_1), int(grid_size_A)))

        # Interpolate
        for i in range(len(assets_reshaped_1)):

            a_prime_points = np.take(A[~np.isnan(assets_reshaped_1[i])],\
            						 np.argsort(assets_reshaped_1[i]))

            Aprime_noadjust_1[i] = interp_as(assets_reshaped_1[i],
                                             a_prime_points,
                                             A, extrap= True)
            Aprime_noadjust_1[i][Aprime_noadjust_1[i] <= 0] = A_min
            Aprime_noadjust_1[i][Aprime_noadjust_1[i] >= A_max_W] = A_max_W
            Aprime_noadjust_1[i][Aprime_noadjust_1[i] == np.nan] = A_min

            c_prime_points = np.take(cons_reshaped_1[i][~np.isnan(
                assets_reshaped_1[i])], np.argsort(assets_reshaped_1[i]))

            C_noadj_1[i] = interp_as(assets_reshaped_1[i],
                                     c_prime_points,
                                     A, extrap= True)
            C_noadj_1[i][C_noadj_1[i] == np.nan] = C_min
            C_noadj_1[i][C_noadj_1[i] <= 0] = C_min
            C_noadj_1[i][C_noadj_1[i] >= C_max] = C_max

            vf_prime_points = np.take(vf_reshaped_1[i][~np.isnan(
                assets_reshaped_1[i])], np.argsort(assets_reshaped_1[i]))

            V_reshaped_1[i] = interp_as(assets_reshaped_1[i],
                                        vf_prime_points,
                                        A)
            V_reshaped_1[i][vf_reshaped_1[i] == np.nan] = 0

            uhdb_prime_points = np.take(uhdb_reshaped_1[i][~np.isnan(
                assets_reshaped_1[i])], np.argsort(assets_reshaped_1[i]))

            UHdb_noadj_1[i] = interp_as(assets_reshaped_1[i],
                                        uhdb_prime_points,
                                        A)
            UHdb_noadj_1[i][UHdb_noadj_1[i] == np.nan] = 1e-250

            uc_points = np.take(uc_reshaped_1[i][~np.isnan(
                assets_reshaped_1[i])], np.argsort(assets_reshaped_1[i]))

            UC_noadj_1[i] = interp_as(assets_reshaped_1[i],
                                      uc_points,
                                      A)
            UC_noadj_1[i][UHdb_noadj_1[i] == np.nan] = 1e-250

        return np.ravel(C_noadj_1), np.ravel(V_reshaped_1),\
            np.ravel(Aprime_noadjust_1), np.ravel(UHdb_noadj_1),\
            np.ravel(UC_noadj_1)

    @njit
    def my_eval_adj_aprime_inv(t, mort_func,
                               UC_prime_func,
                               UC_prime_H_func,
                               UC_prime_M_func,
                               VF,
                               my_X_W_bar_hdjex_ind):
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

        A_prime = np.empty(int(len(my_X_W_bar_hdjex_ind)))
        wealth_bar = np.empty(int(len(my_X_W_bar_hdjex_ind)))
        C = np.copy(A_prime)
        eta_adj = np.copy(A_prime)
        UC_prime_adj = np.copy(A_prime)

        UC_prime_func_ex = UC_prime_func.reshape(all_state_shape)
        UC_prime_H_func_ex = UC_prime_H_func.reshape(all_state_shape)
        V_func_ex = VF.reshape(all_state_shape)
        UC_prime_M_func_ex = UC_prime_M_func.reshape(all_state_shape)

        for i in range(len(my_X_W_bar_hdjex_ind)):

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

            E_ind, alpha_ind, beta_ind = my_X_W_bar_hdjex_ind[i][1],\
                my_X_W_bar_hdjex_ind[i][2],\
                my_X_W_bar_hdjex_ind[i][3]
            Pi_ind, DB_ind, adc_ind = my_X_W_bar_hdjex_ind[i][4], 0,\
                int(my_X_W_bar_hdjex_ind[i][5])

            h_ind, q_ind = int(my_X_W_bar_hdjex_ind[i][6]),\
                int(my_X_W_bar_hdjex_ind[i][7])

            h, q, a_dc = H[h_ind], Q[q_ind], A_DC[adc_ind]

            beta = beta_hat[beta_ind]
           # print(beta)
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

            UC_prime_M_func1D = beta * UC_prime_M_func_ex[DB_ind,
                                                   E_ind, alpha_ind,
                                                   beta_ind, Pi_ind,
                                                   0, adc_ind, h_ind,
                                                   q_ind, :]

            # get function of M_prime as function of A_prime and H_t
            mort_func1D = mort_func[DB_ind, E_ind,
                                    alpha_ind,
                                    beta_ind, Pi_ind,
                                    :, adc_ind, :,
                                    q_ind]

            # Make tuple of args for FOC Euler error functions
            args_HA_FOC = (a_dc, h, q, alpha_housing, beta, mort_func1D,
                           UC_prime_func2D,
                           UC_prime_H_func2D,
                           UC_prime_M_func2D,
                           t)

            args_eval_c = (
                A_min,
                a_dc,
                h,
                q,
                alpha_housing,
                beta,
                mort_func1D,
                UC_prime_func2D,
                UC_prime_H_func2D,
                UC_prime_M_func1D,
                t)

            max_loan = (1 - phi_c)

            # check if there is an interior solution to
            # housing, liq assett and constrained mortgage FOC

            A_prime[i] = np.nan
            C[i] = np.nan
            wealth_bar[i] = np.nan

            if h == H_min:
                A_prime[i] = A_min
                C[i] = C_min
                wealth_bar[i] = C[i] + A_prime[i]\
                    + q * h * ((1 - 0) + tau_housing)\

                eta_adj[i] = u(C[i], h, alpha_housing) + beta\
                    * eval_linear_c(X_cont_WAM2,
                                    VF_func2D,
                                    np.array([A_min, H_min, 0]),
                                    xto.NEAREST)

                UC_prime_adj[i] = uc(C[i], h, alpha_housing)

            elif FOC_housing_adj(A_min, *args_HA_FOC)\
                    * FOC_housing_adj(100, *args_HA_FOC) < 0:
                # if interior solution to a_t+1, calculate it
                A_prime[i] = min(A_max_W,max(brentq(FOC_housing_adj, A_min, 100,
                                        args=args_HA_FOC, disp=False)[0],
                                 A_min))

                C[i] = min(C_max, max(
                    C_min,
                    FOC_housing_adj(
                        A_prime[i],
                        a_dc,
                        h,
                        q,
                        alpha_housing,
                        beta,
                        mort_func1D,
                        UC_prime_func2D,
                        UC_prime_H_func2D,
                        UC_prime_M_func2D,
                        t,
                        ret_cons=True)))
                m_prime_adj = eval_linear_c(
                    X_cont_AH, mort_func1D, np.array([A_prime[i], h]), xto.NEAREST)

                m_prime = max(0, min(m_prime_adj, max_loan))

                wealth_bar[i] = C[i] + A_prime[i]\
                    + q * h * ((1 - m_prime) + tau_housing)\

                eta_adj[i] = u(C[i],
                               h,
                               alpha_housing) + beta * eval_linear_c(X_cont_WAM2,
                                                                     VF_func2D,
                                                                     np.array([A_prime[i],
                                                                               h,
                                                                               m_prime]),
                                                                     xto.NEAREST)
                UC_prime_adj[i] = uc(C[i], h, alpha_housing)

            # if no interior solution with liq asset unconstrainted,
            # check if interior solution to housing and constrained
            # mortage with binding liquid assett FOC at A_min
            elif FOC_housing_adj_bind(C_min, *args_eval_c)\
                    * FOC_housing_adj_bind(C_max, *args_eval_c) < 0:

                results_bq = brentq(FOC_housing_adj_bind, C_min,\
                             C_max, args=args_eval_c, disp=False)
                if results_bq[3] == True:
                    C_at_amin = min(
                        C_max, max(results_bq[0], C_min))


                    uc_prime = uc(C_at_amin, h, alpha_housing)

                    m_prime_at_amin = min(max(0,np.interp(uc_prime, UC_prime_M_func1D, M)),max_loan)

                    UC_prime_RHS_amin = beta * eval_linear_c(X_cont_WAM2,
                                                                     UC_prime_func2D,
                                                                     np.array([A_min,
                                                                               h,
                                                                               m_prime_at_amin]),
                                                                     xto.NEAREST)

                    # if liquid assett const. does not satisfy
                    # FOC, throw point out
                    if uc(C_at_amin, h, alpha_housing) >= UC_prime_RHS_amin:
                        A_prime[i] = A_min
                        C[i] = C_at_amin

                        wealth_bar[i] = C[i] \
                            + A_prime[i]\
                            + q * h * ((1 - m_prime_at_amin) + tau_housing)\

                        eta_adj[i] = u(C[i],
                                       h,
                                       alpha_housing) + beta * eval_linear_c(X_cont_WAM2,
                                                                             VF_func2D,
                                                                             np.array([A_min,
                                                                                       h,
                                                                                       m_prime_at_amin]),
                                                                             xto.NEAREST)
                        UC_prime_adj[i] = uc(C[i], h, alpha_housing)
                    else:
                        pass 

            # we are not force include point with zero housing
            # if solution adjusted to housing
            # less than H_min

            else:
                pass

        return A_prime, C, eta_adj, wealth_bar, UC_prime_adj

    @njit
    def my_eval_noadj_aprime_inv(t,
                                 mort_func,
                                 UC_prime_func,
                                 UC_prime_H_func,
                                 UC_prime_M_func,
                                 VF,
                                 my_X_all_hat_ind):
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

        assets_l = np.zeros(len(my_X_all_hat_ind))
        vf = np.zeros(len(my_X_all_hat_ind))
        cons_l = np.zeros(len(my_X_all_hat_ind))
        uh_db_prime_l = np.zeros(len(my_X_all_hat_ind))
        uc_l = np.zeros(len(my_X_all_hat_ind))

        # Reshape t+1 functions so they can be indexed by current
        # period states and be made into a function of M_t+1
        UC_prime_func_ex = UC_prime_func.reshape(all_state_shape)
        UC_prime_H_func_ex = UC_prime_H_func.reshape(all_state_shape)
        V_func_ex = VF.reshape(all_state_shape)
        UC_prime_M_func_ex = UC_prime_M_func.reshape(all_state_shape)

        for i in range(len(my_X_all_hat_ind)):
            # Loop through each exogenous grid point
            # recall h is H_t i.e. housing *after* depreciation at t
            # i.e. housing that goes into utility at t
            # a_prime is liquid asset taken into next period
            # q is period t price of housing

            h = H[my_X_all_hat_ind[i][7]] * (1 - delta_housing)
            a_prime = A[my_X_all_hat_ind[i][5]]
            q = Q[my_X_all_hat_ind[i][8]]
            m_ind = my_X_all_hat_ind[i][8]
            m = M[my_X_all_hat_ind[i][9]]
            a_dc = A_DC[my_X_all_hat_ind[i][6]]
            m_val = m * h * q

            E_ind = my_X_all_hat_ind[i][1]
            alpha_ind = my_X_all_hat_ind[i][2]
            beta_ind = my_X_all_hat_ind[i][3]
            Pi_ind = my_X_all_hat_ind[i][4]
            DB_ind = 0
            a_ind = int(my_X_all_hat_ind[i][5])
            adc_ind = int(my_X_all_hat_ind[i][6])
            h_ind = int(my_X_all_hat_ind[i][7])
            q_ind = int(my_X_all_hat_ind[i][8])

            beta = beta_hat[beta_ind]
            alpha_housing = alpha_hat[alpha_ind]

            # get the next period U_prime values
            mfunc_ucprime = UC_prime_func_ex[DB_ind,
                                             E_ind, alpha_ind,
                                             beta_ind, Pi_ind,
                                             :, adc_ind, h_ind,
                                             q_ind, :]

            mfunc_ucprime_h = UC_prime_H_func_ex[DB_ind,
                                                 E_ind, alpha_ind,
                                                 beta_ind, Pi_ind,
                                                 :, adc_ind, h_ind,
                                                 q_ind, :]
            mfunc_uc_v_prime = V_func_ex[DB_ind,
                                         E_ind, alpha_ind,
                                         beta_ind, Pi_ind,
                                         :, adc_ind, h_ind,
                                         q_ind, :]

            mfunc_uc_m_prime_a_min = UC_prime_M_func_ex[DB_ind,
                                                  E_ind, alpha_ind,
                                                  beta_ind, Pi_ind,
                                                  0, adc_ind, h_ind,
                                                  q_ind, :]

            mfunc_uc_m_prime = UC_prime_M_func_ex[DB_ind,
                                                  E_ind, alpha_ind,
                                                  beta_ind, Pi_ind,
                                                  :, adc_ind, h_ind,
                                                  q_ind,:]
            mort_func1D = mort_func[DB_ind, E_ind,
                                    alpha_ind,
                                    beta_ind, Pi_ind,
                                    :, adc_ind, h_ind,
                                    q_ind]

            # calculate consumption by checking whether non-negative
            # mortgage consraint binds

            # RHS with full payment
            max_pay_points = np.array([a_prime, 0])
            ucmprime_m_fp = beta * eval_linear_c(X_cont_WAM,
                                                 mfunc_uc_m_prime,
                                                 max_pay_points, xto.NEAREST)
            uc_prime_fp = max(
                1e-250,
                beta *
                eval_linear_c(
                    X_cont_WAM,
                    mfunc_ucprime,
                    max_pay_points,
                    xto.NEAREST))

            # Step 1: check if constrainted by full re-payment of liability
            if a_ind == 0:
                uc_inv_mprime = np.interp(1-phi_c, M, mfunc_uc_m_prime_a_min)
                c_t = min(
                    max(C_min, uc_inv(uc_inv_mprime, h, alpha_housing)), C_max)

                m_prime = (1-phi_c)


            elif uc_prime_fp <= ucmprime_m_fp:
                c_t = min(
                    max(C_min, uc_inv(uc_prime_fp, h, alpha_housing)), C_max)
                m_prime = 0
                UC_prime_RHS = uc_prime_fp

            # Step 2: otherwise, eval unconstrained
            # Note we are still constrained by collateral
            else:
                m_prime_adj = np.interp(a_prime,A, mort_func1D)
                m_prime = max(0, min(m_prime_adj, (1 - phi_c)))
                UC_prime_RHS = max(1e-250,
                                   beta * eval_linear_c(X_cont_WAM,
                                                        mfunc_ucprime,
                                                        np.array([a_prime,
                                                                  m_prime]),
                                                        xto.NEAREST))

                c_t = min(C_max, max(C_min, uc_inv(
                    UC_prime_RHS, h, alpha_housing)))

            mort_payment = m_val - m_prime * h * q
            pts_noadj = np.array([a_prime, m_prime])

            # calculate t+ value function
            V_RHS = beta * eval_linear_c(X_cont_WAM,
                                         mfunc_uc_v_prime,
                                         pts_noadj, xto.NEAREST)

            uh_db_prime_l[i] = max(
                1e-250,
                beta *
                eval_linear_c(
                    X_cont_WAM,
                    mfunc_ucprime_h,
                    pts_noadj,
                    xto.NEAREST))

            # Define the endogenous liq assets after
            # wage and after vol. cont.
            # note we are NOT dividing through by (1+r)
            assets_l[i] = max(0, c_t + a_prime + mort_payment)
            cons_l[i] = c_t
            vf[i] = u(c_t, h, alpha_housing) + V_RHS
            # print(etas[i])
            uc_l[i] = uc(c_t, h, alpha_housing)

        # Re-shape and interpolate the no adjustment endogenous grids
        # we want rows to  index a product of states
        # other than A_prime.
        # vals in the columns are time t liquid asssets for A_t+1
        return cons_l, vf, assets_l, uh_db_prime_l, uc_l

    # Functions that invert the inverse policy function
    # to recover the policy function (EGM step)

    def invert_noadj_prime_pol(assets_noadj_inv,
                               vf_noadj_inv,
                               cons_noadj_inv,
                               uh_db_prime_noadj_inv,
                               uc_prime_noadj_inv):
        """ Inverts the inverse policy function for
                non-adjusters"""

        if comm.rank == 0:

            Aprime_noadjust_1 = np.empty(
                (int(x_all_hat_ind_len / grid_size_A * int(grid_size_A))))
            vf_primes_1 = np.empty((int(x_all_hat_ind_len
                                        / grid_size_A * int(grid_size_A))))
            C_noadj_1 = np.empty((int(x_all_hat_ind_len
                                      / grid_size_A * int(grid_size_A))))
            UHdb_noadj_1 = np.empty((int(x_all_hat_ind_len
                                         / grid_size_A * int(grid_size_A))))
            UC_noadj_1 = np.empty((int(x_all_hat_ind_len
                                       / grid_size_A * int(grid_size_A))))

            assets_reshaped_1 = _noadj_pre_interp_reshape(assets_noadj_inv,
                                                          x_all_hat_ind_len)
            assets_reshaped_1_split = np.array_split(assets_reshaped_1,
                                                     comm.size, axis=0)
            vf_reshaped_1 = _noadj_pre_interp_reshape(vf_noadj_inv,
                                                      x_all_hat_ind_len)
            vf_reshaped_1_split = np.array_split(vf_reshaped_1,
                                                 comm.size, axis=0)

            cons_reshaped_1 = _noadj_pre_interp_reshape(cons_noadj_inv,
                                                        x_all_hat_ind_len)
            cons_reshaped_1_split = np.array_split(cons_reshaped_1,
                                                   comm.size, axis=0)
            uhdb_reshaped_1 = _noadj_pre_interp_reshape(uh_db_prime_noadj_inv,
                                                        x_all_hat_ind_len)
            uhdb_reshaped_1_split = np.array_split(uhdb_reshaped_1,
                                                   comm.size, axis=0)
            uc_reshaped_1 = _noadj_pre_interp_reshape(uc_prime_noadj_inv,
                                                      x_all_hat_ind_len)
            uc_reshaped_1_split = np.array_split(uc_reshaped_1,
                                                 comm.size, axis=0)
        else:
            uc_reshaped_1_split = None
            uhdb_reshaped_1_split = None
            vf_reshaped_1_split = None
            assets_reshaped_1_split = None
            cons_reshaped_1_split = None

            Aprime_noadjust_1 = None
            vf_primes_1 = None
            C_noadj_1 = None
            UHdb_noadj_1 = None
            UC_noadj_1 = None

        my_uc_reshaped_1 = comm.scatter(uc_reshaped_1_split, root=0)
        my_uhdb_reshaped_1 = comm.scatter(uhdb_reshaped_1_split, root=0)
        my_vf_reshaped_1 = comm.scatter(vf_reshaped_1_split, root=0)
        my_assets_reshaped_1 = comm.scatter(assets_reshaped_1_split, root=0)
        my_cons_reshaped_1 = comm.scatter(cons_reshaped_1_split, root=0)

        my_no_adj_pols_out_list = my_invert_noadj_prime_pol(
            my_assets_reshaped_1,
            my_vf_reshaped_1,
            my_cons_reshaped_1,
            my_uhdb_reshaped_1,
            my_uc_reshaped_1)

        my_C_noadj_1, my_vf_primes_1, my_Aprime_noadjust_1,\
            my_UHdb_noadj_1, my_UC_noadj_1 = my_no_adj_pols_out_list

        sendcounts = np.array(comm.gather(len(my_C_noadj_1), 0))

        comm.Gatherv(np.ascontiguousarray(my_C_noadj_1),
                     recvbuf=(C_noadj_1, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_vf_primes_1),
                     recvbuf=(vf_primes_1, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_Aprime_noadjust_1),
                     recvbuf=(Aprime_noadjust_1, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_UHdb_noadj_1),
                     recvbuf=(UHdb_noadj_1, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_UC_noadj_1),
                     recvbuf=(UC_noadj_1, sendcounts), root=0)

        del my_uc_reshaped_1, my_uhdb_reshaped_1, my_vf_reshaped_1,\
            my_assets_reshaped_1, my_cons_reshaped_1
        del my_C_noadj_1, my_vf_primes_1, my_Aprime_noadjust_1,\
            my_UHdb_noadj_1, my_UC_noadj_1
        del uc_reshaped_1_split, uhdb_reshaped_1_split, vf_reshaped_1_split,\
            assets_reshaped_1_split, cons_reshaped_1_split

        if comm.rank == 0:

            Aprime_noadjust_1 = Aprime_noadjust_1\
                .reshape(all_state_A_last_shape)
            vf_primes_1 = vf_primes_1\
                .reshape(all_state_A_last_shape)
            C_noadj_1 = C_noadj_1\
                .reshape(all_state_A_last_shape)
            UHdb_noadj_1 = UHdb_noadj_1\
                .reshape(all_state_A_last_shape)
            UC_noadj_1 = UC_noadj_1\
                .reshape(all_state_A_last_shape)

            Aprime_noadj = np.transpose(Aprime_noadjust_1,
                                        (0, 1, 2, 3, 4, 9, 5, 6, 7, 8))
            vf_noadj = np.transpose(vf_primes_1,
                                    (0, 1, 2, 3, 4, 9, 5, 6, 7, 8))
            C_noadj = np.transpose(C_noadj_1,
                                   (0, 1, 2, 3, 4, 9, 5, 6, 7, 8))
            UHdb_noadj = np.transpose(UHdb_noadj_1,
                                      (0, 1, 2, 3, 4, 9, 5, 6, 7, 8))
            UC_noadj = np.transpose(UC_noadj_1,
                                    (0, 1, 2, 3, 4, 9, 5, 6, 7, 8))

            return C_noadj, vf_noadj, Aprime_noadj, UHdb_noadj, UC_noadj

        else:
            return None, None, None, None, None

    def invert_adj_prime_pol(a_adj_inv,
                             c_adj_inv,
                             vf_adj_inv,
                             wealth_adj_inv,
                             uc_prime_adj_inv,
                             comm):
        """ Inverts the inverse policy function for
                housing adjusters"""

        if comm.rank == 0:

            assets_prime_adj_1 \
                = np.empty((int(len_X_W_bar_hdjex_ind / grid_size_H) *
                            int(grid_size_A)))
            H_prime_adj_1\
                = np.empty((int(len_X_W_bar_hdjex_ind / grid_size_H) *
                            int(grid_size_A)))
            c_prime_adj_1\
                = np.empty((int(len_X_W_bar_hdjex_ind / grid_size_H) *
                            int(grid_size_A)))
            vf_adj_prime_1\
                = np.empty((int(len_X_W_bar_hdjex_ind / grid_size_H) *
                            int(grid_size_A)))
            uc_prime_adj_1\
                = np.empty((int(len_X_W_bar_hdjex_ind / grid_size_H) *
                            int(grid_size_A)))
        else:
            assets_prime_adj_1 = None
            H_prime_adj_1 = None
            c_prime_adj_1 = None
            vf_adj_prime_1 = None
            uc_prime_adj_1 = None

        if comm.rank == 0:

            A_prime_adj_reshape = reshape_make_h_last(a_adj_inv)
            C_adj_reshape = reshape_make_h_last(c_adj_inv)
            vf_adj_reshape = reshape_make_h_last(vf_adj_inv)
            wealth_adj_reshape = reshape_make_h_last(wealth_adj_inv)
            uc_prime_adj_reshape = reshape_make_h_last(uc_prime_adj_inv)

            A_adj_reshape_split = np.array_split(A_prime_adj_reshape,
                                                 comm.size, axis=0)
            C_adj_split = np.array_split(C_adj_reshape, comm.size, axis=0)
            vf_adj_reshape_split = np.array_split(vf_adj_reshape, comm.size,
                                                  axis=0)
            wealth_reshape_split = np.array_split(wealth_adj_reshape,
                                                  comm.size,
                                                  axis=0)
            uc_prime_adj_reshape_split = np.array_split(uc_prime_adj_reshape,
                                                        comm.size,
                                                        axis=0)

        else:
            A_adj_reshape_split = None
            C_adj_split = None
            vf_adj_reshape_split = None
            wealth_reshape_split = None
            m_prime_adj_reshape_split = None
            uc_prime_adj_reshape_split = None

        my_A_adj_reshape = comm.scatter(A_adj_reshape_split, root=0)
        my_C_adj = comm.scatter(C_adj_split, root=0)
        my_vf_adj_reshape = comm.scatter(vf_adj_reshape_split, root=0)
        my_wealth_reshape = comm.scatter(wealth_reshape_split, root=0)
        my_uc_prime_adj_reshape = comm.scatter(
            uc_prime_adj_reshape_split, root=0)

        my_out_adj_funcs = my_invert_adj_prime_pol(my_wealth_reshape,
                                                   my_A_adj_reshape,
                                                   my_C_adj,
                                                   my_vf_adj_reshape,
                                                   my_uc_prime_adj_reshape)

        my_assets_prime_adj_1, my_H_prime_adj_1,\
            my_c_prime_adj_1, my_vf_adj_prime_1, my_uc_prime_adj_1 = my_out_adj_funcs

        sendcounts = np.array(comm.gather(len(my_assets_prime_adj_1), 0))

        comm.Gatherv(np.ascontiguousarray(my_assets_prime_adj_1),
                     recvbuf=(assets_prime_adj_1, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_H_prime_adj_1),
                     recvbuf=(H_prime_adj_1, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_c_prime_adj_1),
                     recvbuf=(c_prime_adj_1, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_vf_adj_prime_1),
                     recvbuf=(vf_adj_prime_1, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_uc_prime_adj_1),
                     recvbuf=(uc_prime_adj_1, sendcounts), root=0)

        if comm.rank == 0:

            H_adj = _rehape_adj_post_interp(H_prime_adj_1)
            C_adj = _rehape_adj_post_interp(c_prime_adj_1)
            A_adj = _rehape_adj_post_interp(assets_prime_adj_1)
            VF_adj = _rehape_adj_post_interp(vf_adj_prime_1)
            UC_prime_adj = _rehape_adj_post_interp(uc_prime_adj_1)

            return C_adj, H_adj, A_adj, VF_adj, UC_prime_adj
        else:
            return None

    @njit
    def my_invert_adj_prime_pol(wealth_bar_reshape,
                                A_prime_adj_reshape,
                                C_adj_reshape,
                                eta_adj_reshape,
                                uc_prime_adj_reshape):
        """ Invert adjuster inverse policy functions on each worker via
                interpolation"""

        # New pol funcs which will be interpolated over uniform
        # wealth grid
        assets_prime_adj_inverted \
            = np.empty((len(wealth_bar_reshape),
                        int(grid_size_A)))
        H_prime_adj_inverted\
            = np.empty((len(wealth_bar_reshape),
                        int(grid_size_A)))
        c_prime_adj_inverted\
            = np.empty((len(wealth_bar_reshape),
                        int(grid_size_A)))
        eta_adj_prime_inverted\
            = np.empty((len(wealth_bar_reshape),
                        int(grid_size_A)))
        uc_prime_adj_prime_inverted\
            = np.empty((len(wealth_bar_reshape),
                        int(grid_size_A)))

        for i in range(len(wealth_bar_reshape)):

            # Take the values at which a inverse solution exists and sort
            wealth_x = wealth_bar_reshape[i][~np.isnan(wealth_bar_reshape[i])]
            assets_clean = A_prime_adj_reshape[i][~np.isnan(
                wealth_bar_reshape[i])]
            assts_x = np.take(assets_clean, np.argsort(wealth_x))

            h_clean = H_R[~np.isnan(wealth_bar_reshape[i])]
            h_x = np.take(h_clean, np.argsort(wealth_x))

            c_clean = C_adj_reshape[i][~np.isnan(wealth_bar_reshape[i])]
            c_x = np.take(c_clean, np.argsort(wealth_x))

            eta_clean = eta_adj_reshape[i][~np.isnan(wealth_bar_reshape[i])]
            eta_x = np.take(eta_clean, np.argsort(wealth_x))

            uc_clean = uc_prime_adj_reshape[i][~np.isnan(
                wealth_bar_reshape[i])]
            uc_x = np.take(uc_clean, np.argsort(wealth_x))

            wealth_xs = np.sort(wealth_x)

            # Invert by interpolation
            assets_prime_adj_inverted[i] = interp_as(wealth_xs, assts_x, W_W, extrap= True)
            assets_prime_adj_inverted[i][assets_prime_adj_inverted[i] <= 0]\
                = A_min
            assets_prime_adj_inverted[i][assets_prime_adj_inverted[i]
                                         == np.nan] = A_min
            assets_prime_adj_inverted[i][assets_prime_adj_inverted[i] >= A_max_W] = A_max_W

            c_prime_adj_inverted[i] = interp_as(wealth_xs, c_x, W_W, extrap= True)
            c_prime_adj_inverted[i][c_prime_adj_inverted[i] <= 0] = C_min
            c_prime_adj_inverted[i][c_prime_adj_inverted[i] == np.nan] = C_min
            c_prime_adj_inverted[i][c_prime_adj_inverted[i] >= C_max] = C_max

            H_prime_adj_inverted[i] = interp_as(wealth_xs, h_x, W_W, extrap= True)
            H_prime_adj_inverted[i][H_prime_adj_inverted[i] <= 0] = H_min
            H_prime_adj_inverted[i][H_prime_adj_inverted[i] == np.nan] = H_min
            H_prime_adj_inverted[i][H_prime_adj_inverted[i] >= H_max] = H_max

            eta_adj_prime_inverted[i] = interp_as(wealth_xs, eta_x, W_W)
            uc_prime_adj_prime_inverted[i] = interp_as(wealth_xs, uc_x, W_W)
            #print(assts_x)
            #print(eta_adj_prime_inverted[i])

        return np.ravel(assets_prime_adj_inverted),\
            np.ravel(H_prime_adj_inverted),\
            np.ravel(c_prime_adj_inverted),\
            np.ravel(eta_adj_prime_inverted),\
            np.ravel(uc_prime_adj_prime_inverted)

    # Define functions that produce the policy functions and
    # evaluate the marginal utility and value function
    # on the full state-space

    def gen_rhs_euler_pols_noadj(t,
                                 comm,
                                 mort_func,
                                 UC_prime,
                                 UC_prime_H,
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

        if comm.rank == 0:
            cons_l = np.empty(x_all_hat_ind_len)
            vf_noadj_inv = np.empty(x_all_hat_ind_len)
            assets_l = np.empty(x_all_hat_ind_len)
            uh_db_prime_l = np.empty(x_all_hat_ind_len)
            uc_prime_l = np.empty(x_all_hat_ind_len)
        else:
            cons_l = None
            vf_noadj_inv = None
            assets_l = None
            uh_db_prime_l = None
            uc_prime_l = None

        my_cons_l, my_vf_noadj_inv, my_assets_l, my_uh_db_prime_l,\
            my_uc_prime_l = my_eval_noadj_aprime_inv(t, mort_func,
                                                     UC_prime, UC_prime_H, UC_prime_M,
                                                     VF, my_X_all_hat_ind)

        sendcounts = np.array(comm.gather(len(my_cons_l), 0))

        comm.Gatherv(np.ascontiguousarray(my_cons_l),
                     recvbuf=(cons_l, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_vf_noadj_inv),
                     recvbuf=(vf_noadj_inv, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_assets_l),
                     recvbuf=(assets_l, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_uh_db_prime_l),
                     recvbuf=(uh_db_prime_l, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_uc_prime_l),
                     recvbuf=(uc_prime_l, sendcounts), root=0)

        C_noadj, vf_noadj, Aprime_noadj, UH_dp_prime_noadj, UC_prime_noadj\
            = invert_noadj_prime_pol(assets_l, vf_noadj_inv,
                                     cons_l, uh_db_prime_l, uc_prime_l)

        if comm.rank == 0:
            # Interpolate on endogenous grid
            # Re-shape policy functions
            noadj_vals = gen_rhs_eval_pol_points_noadj(
                reshape_nadj_RHS(C_noadj),
                reshape_nadj_RHS(vf_noadj),
                reshape_nadj_RHS(Aprime_noadj),
                reshape_nadj_RHS(UH_dp_prime_noadj),
                reshape_nadj_RHS(UC_prime_noadj),
                points_noadj)

            noadj_vals = noadj_vals.reshape((1, grid_size_W,
                                             grid_size_alpha,
                                             grid_size_beta,
                                             len(Pi),
                                             grid_size_H,
                                             grid_size_Q,
                                             grid_size_M,
                                             len(V),
                                             grid_size_A,
                                             grid_size_DC,
                                             3))\
                .transpose((0, 1, 2, 3, 8, 4, 9, 10, 5, 6, 7, 11))\
                .reshape((x_all_ind_len, 3))

            return noadj_vals, C_noadj, vf_noadj, Aprime_noadj
        else:
            return None

    def gen_rhs_euler_pols_adj(t, comm,
                               mort_func,
                               UC_prime,
                               UC_prime_H,
                               UC_prime_M,
                               VF,
                               points_adj):
        """Function evaluates polices, then calls on function to create RHS
        interpoled policies and returns both."""

        if comm.rank == 0:
            A_prime_inv = np.empty(len_X_W_bar_hdjex_ind)
            C_inv = np.empty(len_X_W_bar_hdjex_ind)
            vf_adj_inv = np.empty(len_X_W_bar_hdjex_ind)
            wealth_bar = np.empty(len_X_W_bar_hdjex_ind)
            uc_prime_adj_inv = np.empty(len_X_W_bar_hdjex_ind)
        else:
            A_prime_inv = None
            C_inv = None
            vf_adj_inv = None
            wealth_bar = None
            uc_prime_adj_inv = None

        my_A_prime_inv, my_C_inv, my_vf_adj_inv, my_wealth_bar_inv,\
            my_uc_prime_adj_inv = my_eval_adj_aprime_inv(t,
                                                         mort_func,
                                                         UC_prime,
                                                         UC_prime_H,
                                                         UC_prime_M, VF,
                                                         my_X_W_bar_hdjex_ind)

        sendcounts = np.array(comm.gather(len(my_A_prime_inv), 0))

        comm.Gatherv(np.ascontiguousarray(my_A_prime_inv),
                     recvbuf=(A_prime_inv, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_C_inv),
                     recvbuf=(C_inv, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_vf_adj_inv),
                     recvbuf=(vf_adj_inv, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_wealth_bar_inv),
                     recvbuf=(wealth_bar, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_uc_prime_adj_inv),
                     recvbuf=(uc_prime_adj_inv, sendcounts), root=0)

        out_interps = invert_adj_prime_pol(A_prime_inv, C_inv,
                                           vf_adj_inv, wealth_bar,
                                           uc_prime_adj_inv, comm)

        if comm.rank == 0:
            C_adj, H_adj, Aprime_adj, vf_adj, uc_prime_adj = out_interps
        else:
            H_adj = None
            vf_adj = None
            uc_prime_adj = None

        adj_vals = gen_rhs_adj_eval_points(vf_adj,
                                           uc_prime_adj,
                                           points_adj, comm)
        if comm.rank == 0:

            adj_vals = adj_vals.reshape((1, grid_size_W,
                                         grid_size_alpha,
                                         grid_size_beta, len(Pi),
                                         grid_size_Q, grid_size_M,
                                         len(V),
                                         grid_size_A, grid_size_DC,
                                         grid_size_H, 2))\
                .transpose((0, 1, 2, 3, 7, 4, 8, 9, 10, 5, 6, 11))\
                .reshape((x_all_ind_len, 2))
            #print(adj_vals[:,0])
            return adj_vals, C_adj, H_adj, Aprime_adj
        else:
            return None

    def gen_rhs_rent_euler_pols(UC_prime, VF_prime, points_rent):

        H_rent, V_rent, UC_rent = eval_rent_pol_W(UC_prime, VF_prime)
        rent_pols = reshape_rent_RHS(H_rent)
        rent_VF = reshape_rent_RHS(V_rent)
        rent_UC = reshape_rent_RHS(UC_rent)
        rent_vals = gen_rhs_eval_pol_points_rent(rent_UC, rent_VF, points_rent)
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

    def gen_rhs_adj_eval_points(Etaprime_adj,
                                uc_prime_adj,
                                points_adj,
                                comm):
        """Evalutes policies for housing adjusters on all points in the full
        state-space grid."""

        # Empty adjuster policy grid
        if comm.rank == 0:

            Etaprime_adj = reshape_adj_RHS(Etaprime_adj)
            uc_prime_adj = reshape_adj_RHS(uc_prime_adj)

            etaadj_eval = np.empty(int(len(Etaprime_adj)
                                       * grid_size_H
                                       * len(V)
                                       * grid_size_A
                                       * grid_size_DC))
            ucadj_eval = np.empty(int(len(Etaprime_adj)
                                      * grid_size_H
                                      * len(V)
                                      * grid_size_A
                                      * grid_size_DC))
        else:
            adj_vals_small1 = None
            hadj_eval = None
            etaadj_eval = None
            ucadj_eval = None

        if comm.rank == 0:
            Etaprime_adj_split = np.array_split(
                Etaprime_adj, comm.size, axis=0)
            uc_prime_adj_split = np.array_split(
                uc_prime_adj, comm.size, axis=0)
            points_adj_split = np.array_split(points_adj, comm.size, axis=0)

        else:
            Etaprime_adj_split = None
            uc_prime_adj_split = None
            points_adj_split = None

        my_Etaprime_adj = comm.scatter(Etaprime_adj_split, root=0)
        my_uc_prime_adj = comm.scatter(uc_prime_adj_split, root=0)
        my_points_adj = comm.scatter(points_adj_split, root=0)

        my_etaadj_eval1, my_ucadj_eval1 \
            = my_gen_rhs_eval_pol_adj(my_Etaprime_adj,
                                      my_uc_prime_adj, my_points_adj)

        sendcounts = np.array(comm.gather(len(my_etaadj_eval1), 0))

        comm.Gatherv(np.ascontiguousarray(my_etaadj_eval1),
                     recvbuf=(etaadj_eval, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_ucadj_eval1),
                     recvbuf=(ucadj_eval, sendcounts), root=0)

        if comm.rank == 0:

            adj_vals_small1 = np.column_stack((etaadj_eval, ucadj_eval))
            adj_vals_small2 = adj_vals_small1.reshape(len(Etaprime_adj),
                                                      int(grid_size_H
                                                          * len(V)
                                                          * grid_size_A
                                                          * grid_size_DC), 2)
            return adj_vals_small2
        else:
            return None

    @njit
    def my_gen_rhs_eval_pol_adj(my_Etaprime_adj, my_uc_prime_adj,
                                my_points_adj):

        my_etaadj_eval = np.empty((len(my_Etaprime_adj),
                                   int(grid_size_H * len(V)
                                       * grid_size_A
                                       * grid_size_DC)))

        my_ucadj_eval = np.empty((len(my_Etaprime_adj),
                                  int(grid_size_H * len(V)
                                      * grid_size_A
                                      * grid_size_DC)))

        for i in range(len(my_Etaprime_adj)):
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

            my_etaadj_eval[i, :] = eval_linear_c(X_QH_W, my_Etaprime_adj[i],
                                                 my_points_adj[i], xto.NEAREST)
            my_ucadj_eval[i, :] = eval_linear_c(X_QH_W, my_uc_prime_adj[i],
                                                my_points_adj[i], xto.NEAREST)

        
        my_etaadj_eval1 = np.ravel(my_etaadj_eval)
        my_ucadj_eval1 = np.ravel(my_ucadj_eval)

        return my_etaadj_eval1, my_ucadj_eval1

    @njit
    def gen_rhs_eval_pol_points_noadj(C_noadj,
                                      etas_noadj,
                                      Aprime_noadj,
                                      UHDBprime_noadj,
                                      UCprime_noadj,
                                      points_noadj):
        """Evalutes pol for housing non-adjusters on all points in the full
        state-space grid."""

        # Empty no-adjuster policy array
        noadj_vals_small1 = np.empty((len(C_noadj),
                                      int(len(V) * grid_size_A
                                          * grid_size_DC),
                                      3))

        for i in prange(len(C_noadj)):

            noadj_vals_small1[i, :, 0] = eval_linear_c(
                X_cont_W_hat, etas_noadj[i], points_noadj[i], xto.NEAREST)

            noadj_vals_small1[i, :, 1] = eval_linear_c(
                X_cont_W_hat, UHDBprime_noadj[i], points_noadj[i], xto.NEAREST)
            noadj_vals_small1[i, :, 2] = eval_linear_c(
                X_cont_W_hat, UCprime_noadj[i], points_noadj[i], xto.NEAREST)

        return noadj_vals_small1

    @njit
    def gen_rhs_eval_pol_points_rent(H_rent, VF_rent, points_rent):
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
            rent_vals_small1[i, :, 0] = eval_linear_c(
                X_QH_W, H_rent[i], points_rent[i], xto.NEAREST)
            rent_vals_small1[i, :, 1] = eval_linear_c(
                X_QH_W, VF_rent[i], points_rent[i], xto.NEAREST)

        return rent_vals_small1

    # Function to combine policices over non-stochastic discrete choices

    def comb_rhs_pol_rent_norent(t, noadj_vals,
                                 adj_vals,
                                 rent_vals,
                                 my_X_all_ind,
                                 A_prime):
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

        h_vals = H[my_X_all_ind[:, 8]] * (1 - delta_housing)
        pi_list = Pi[my_X_all_ind[:, 5]]
        v_list = V[my_X_all_ind[:, 4]]
        Q_list = Q[my_X_all_ind[:, 9]]

        # Generate indicator function for non-adjuster (|eta|<=1)
        etas_noadj = noadj_vals[:, 0]
        etas_adj = adj_vals[:, 0]
        etas_ind = etas_noadj > etas_adj
  
        eta_mu = etas_noadj / etas_adj

        # Marginal bequest value if person diese at start of period
        b_aprime = b_prime(A_prime)

        # Value function of non-renter and renter
        vf_no_rent = (etas_ind * noadj_vals[:,
                                           0] + (1 - etas_ind) * adj_vals[:, 0])
        vf_prime_rent = rent_vals[:, 1]
        renter_ind = vf_prime_rent > vf_no_rent
        zeta = vf_prime_rent / vf_no_rent
        #print(vf_no_rent)

        # Generate policies for non-renters
        # eta*no-adjust + (1-eta)*adjust
        # recall final 5th col of adj_vals is
        # A_DC prime vals (common for renters, adjusters \
        # and non-adjusters since it only depends on V_t
        #h_prime_norent_vals = adj_vals[:, 0]
        #alpha_housing_list = alpha_hat[my_X_all_ind[:, 2]]

        # Generate  adj_value functions for stochastic discrete choices
        v_ind = v_list > 0
        pi_ind = np.zeros(len(pi_list))
        pi_ind[pi_list != def_pi] = 1
        vol_cost = adj_v(t, A[my_X_all_ind[:, 6]]) * v_ind
        # check whether prime value or what goes in to ADC
        pi_cost = adj_pi(t, A_DC[my_X_all_ind[:, 7]], adj_p(17)) * pi_ind
        #print(pi_cost)

        Xi_norent = vf_no_rent - vol_cost - pi_cost
        Xi_rent = vf_prime_rent - vol_cost - pi_cost

        # Recall 2nd col. of rent vals is the cash at hand for renters
        # at t, thus A_t+1 = cash at hand - rent payments - consumption
        #a_prime_rent = rent_vals[:, 1] - c_prime_rent \
        #   - h_prime_rent * phi_r * Q[my_X_all_ind[:, 9]]
        #a_prime_rent[a_prime_rent <= A_min] = A_min

        # Marginal utilis of consumption for renters and non-renters
        uc_adj = adj_vals[:, 1]
        uc_noadj = noadj_vals[:, 2]
        ucfunc_norent = etas_ind * uc_noadj + (1 - etas_ind) * uc_adj
        ucfunc_rent = rent_vals[:, 0]

        # Start of current period marginal price of housing for renters
        uc_h_prime_rent_adj = Q_list * (1 - delta_housing) * ucfunc_rent

        # start of current period marginal price of housing for adjusters
        uc_h_prime_own_adj = Q_list * (1 - delta_housing) * uc_adj

        # Start of current period marginal price of housing for non-adjusters
        uc_h_prime_rent_noadj = ((1 - delta_housing)**2) * noadj_vals[:, 1]

        # Combine marginal utils and value fiunction across renters,
        # non-renters

        uc_prime = s[int(t - 1)] * (1 + r) * (renter_ind * ucfunc_rent
                                              + (1 - renter_ind) * ucfunc_norent)\
            + (1 + r) * (1 - s[int(t - 1)]) * b_aprime

        uc_prime_h = s[int(t - 1)] * (renter_ind * uc_h_prime_rent_adj
                                      + (1 - renter_ind) * (etas_ind * uc_h_prime_rent_noadj
                                                            + (1 - etas_ind) * uc_h_prime_own_adj))\
            + (1 - s[int(t - 1)]) * Q_list * b_aprime * (1 - delta_housing)

        uc_prime_m = uc_prime / (1 + r)

        vf_prime = s[int(t - 1)] * (renter_ind * vf_prime_rent + (1 - renter_ind) * vf_no_rent) +\
                    (1-s[int(t - 1)])*b(A_prime)
        Xi = renter_ind * Xi_rent + (1 - renter_ind) * Xi_norent

        return vf_prime, Xi, zeta, eta_mu, uc_prime, uc_prime_h, uc_prime_m

    def gen_rhs_uc_func(comm, t,
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

        # Scatter adj, noadj and rent grid vals to all cores

        if comm.rank == 0:
            noadj_vals_chunks = np.array_split(noadj_vals, comm.size, axis=0)
            adj_vals_chunks = np.array_split(adj_vals, comm.size, axis=0)
            rent_vals_chunks = np.array_split(rent_vals, comm.size, axis=0)
            A_prime_split = np.array_split(A_prime, comm.size, axis=0)
        else:
            noadj_vals_chunks = None
            adj_vals_chunks = None
            rent_vals_chunks = None
            A_prime_split = None

        if comm.rank == 0:

            xi = np.empty(len(noadj_vals))
            zeta = np.empty(len(noadj_vals))
            eta_mu = np.empty(len(noadj_vals))
            uc_prime_m = np.empty(len(noadj_vals))
            vf = np.empty(len(noadj_vals))
            uc_prime_h = np.empty(len(noadj_vals))
            uc_prime = np.empty(len(noadj_vals))
        else:
            xi = None
            zeta = None
            uc_prime_m = None
            vf = None
            uc_prime_h = None
            uc_prime = None
            eta_mu = None

        my_noadj_vals = comm.scatter(noadj_vals_chunks, root=0)
        my_adj_vals = comm.scatter(adj_vals_chunks, root=0)
        my_rent_vals = comm.scatter(rent_vals_chunks, root=0)
        my_A_prime = comm.scatter(A_prime_split, root=0)

        my_vf,\
            my_xi,\
            my_zeta,\
            my_eta_mu,\
            my_uc_prime,\
            my_uc_prime_h,\
            my_uc_prime_m, = comb_rhs_pol_rent_norent(t, my_noadj_vals,
                                                      my_adj_vals,
                                                      my_rent_vals,
                                                      my_X_all_ind,
                                                      my_A_prime)

        sendcounts = np.array(comm.gather(len(my_uc_prime_m), 0))

        comm.Gatherv(np.ascontiguousarray(my_uc_prime),
                     recvbuf=(uc_prime, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_xi),
                     recvbuf=(xi, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_eta_mu),
                     recvbuf=(eta_mu, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_uc_prime_h),
                     recvbuf=(uc_prime_h, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_zeta),
                     recvbuf=(zeta, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_uc_prime_m),
                     recvbuf=(uc_prime_m, sendcounts), root=0)
        comm.Gatherv(np.ascontiguousarray(my_vf),
                     recvbuf=(vf, sendcounts), root=0)

        del my_noadj_vals, my_adj_vals, my_rent_vals, my_A_prime
        del my_uc_prime, my_xi, my_uc_prime_h, my_zeta, my_vf
        gc.collect()

        if comm.rank == 0:

            UC_prime_B = uc_prime
            UC_prime_M_B = uc_prime_m
            UC_prime_H_B = uc_prime_h
            VF_B = vf
            Xi = xi

            return UC_prime_B, UC_prime_H_B, UC_prime_M_B,\
                VF_B, Xi,\
                zeta.astype(np.float32).reshape(all_state_shape_hat),\
                eta_mu.reshape(all_state_shape_hat)

        else:
            return 0

    # Functions to generate and condition out the discrete
    # choice probabilities

    def _gen_probs(my_xi, scaling_param):

        """Generates discrete choice probabilities
            from value function and scaling param"""

        my_Xi_copi_temp = np.add(my_xi / scaling_param,
                                 - np.max(my_xi / scaling_param,
                                          axis=1)[:, np.newaxis])
        my_prob_pi = np.exp(my_Xi_copi_temp)\
            / np.sum(np.exp(my_Xi_copi_temp), axis=1)[:, np.newaxis]

        my_prob_pi[np.where(np.isnan(my_prob_pi))] = .001


        return np.ravel(my_prob_pi)

    def condition_DC(UC_prime_B,
                     UC_prime_H_B,
                     UC_prime_M_B, VF_B,
                     Xi, comm):
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

        if comm.rank == 0:

            UC_prime_copi, UC_prime_H_copi,\
                UC_prime_M_copi, VF_B_copi, Xi_copi = \
                reshape_out_pi(UC_prime_B),\
                reshape_out_pi(UC_prime_H_B),\
                reshape_out_pi(UC_prime_M_B), reshape_out_pi(VF_B),\
                reshape_out_pi(Xi)

            Xi_copi_split = np.split(Xi_copi, comm.size, axis=0)
            prob_pi = np.empty(len(Xi_copi) * len(Pi))

        else:
            Xi_copi_split = None
            prob_pi = None
            scaling_pi_split = None
            UC_prime_copi = None
            UC_prime_H_copi = None
            UC_prime_M_copi = None
            VF_B_copi = None
            Xi_copi = None

        scaling_pi = acc_ind * sigma_DC_pi + (1 - acc_ind) * sigma_DB_pi
        my_xi_copi = comm.scatter(Xi_copi_split, root=0)
        my_prob_pi = _gen_probs(my_xi_copi, scaling_pi)

        sendcounts = np.array(comm.gather(len(my_prob_pi), 0))

        comm.Gatherv(np.ascontiguousarray(my_prob_pi),
                     recvbuf=(prob_pi, sendcounts), root=0)
        if comm.rank == 0:
            prob_pi = prob_pi.reshape((len(Xi_copi), len(Pi)))
            

        else:
            prob_pi = None

        if comm.rank == 0:
            UC_prime_cpi = _my_einsum_row(prob_pi, UC_prime_copi)

            UC_prime_H_cpi = _my_einsum_row(prob_pi, UC_prime_H_copi)
            UC_prime_M_cpi = _my_einsum_row(prob_pi, UC_prime_M_copi)
            VF_B_cpi = _my_einsum_row(prob_pi, VF_B_copi)
            Xi_cpi = _my_einsum_row(prob_pi, Xi_copi)


        if comm.rank == 0:
            # Reshape to condition out V
            UC_prime_cov, UC_prime_H_cov,\
                UC_prime_M_cov, VF_B_cov, Xi_cov\
                = reshape_out_V(UC_prime_cpi),\
                reshape_out_V(UC_prime_H_cpi),\
                reshape_out_V(UC_prime_M_cpi),\
                reshape_out_V(VF_B_cpi),\
                reshape_out_V(Xi_cpi)

            prob_v = np.empty(len(Xi_cov) * len(V))
            Xi_cov_split = np.split(Xi_cov, comm.size, axis=0)

        else:
            Xi_cov_split = None
            prob_v = None
            scaling_v_split = None
            UC_prime_cov = None
            UC_prime_H_cov = None
            UC_prime_M_cov = None
            VF_B_cov = None
            Xi_cov = None

        scaling_v = acc_ind * sigma_DC_V + (1 - acc_ind) * sigma_DB_V
        my_xi_cov = comm.scatter(Xi_cov_split, root=0)
        my_prob_cov = _gen_probs(my_xi_cov, scaling_v)

        sendcounts = np.array(comm.gather(len(my_prob_cov), 0))

        comm.Gatherv(np.ascontiguousarray(my_prob_cov),
                     recvbuf=(prob_v, sendcounts), root=0)

        if comm.rank == 0:
            prob_v = prob_v.reshape((len(Xi_cov), len(V)))
            #print(prob_v)

        else:
            prob_v = None

        if comm.rank == 0:
            UC_prime_cv = _my_einsum_row(prob_v, UC_prime_cov)
            UC_prime_H_cv = _my_einsum_row(prob_v, UC_prime_H_cov)
            UC_prime_M_cv = _my_einsum_row(prob_v, UC_prime_M_cov)
            VF_B_cv = _my_einsum_row(prob_v, VF_B_cov)

        if comm.rank == 0:
            UC_prime_DCC, UC_prime_H_DCC, UC_prime_M_DCC, VF_DCC\
                = reshape_X_bar(UC_prime_cv),\
                reshape_X_bar(UC_prime_H_cv),\
                reshape_X_bar(UC_prime_M_cv),\
                reshape_X_bar(VF_B_cv)

            return UC_prime_DCC, UC_prime_H_DCC, UC_prime_M_DCC, VF_DCC,\
                prob_v.reshape(prob_v_shape), prob_pi.reshape(prob_pi_shape)
        else:
            return None, None, None, None, None, None

    # Function that conditions all exogenous shocks

    @njit
    def UC_cond_all(t, UC_prime_DCC,
                    UC_prime_H_DCC,
                    UC_prime_M_DCC,
                    VF_DCC,
                    my_X_all_ind):
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
        my_UC_prime = np.empty(len(my_X_all_ind))
        my_UC_prime_H = np.empty(len(my_X_all_ind))
        my_UC_prime_M = np.empty(len(my_X_all_ind))
        my_VF = np.empty(len(my_X_all_ind))\

        #print(VF_DCC)

        for i in range(len(my_X_all_ind)):
            # print(my_X_all_ind[i][1])

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
            E_ind = my_X_all_ind[i, 1]
            alpha_ind = my_X_all_ind[i, 2]
            beta_ind = my_X_all_ind[i, 3]
            DB_ind = int(0)
            pi_ind = my_X_all_ind[i, 4]
            a_ind = my_X_all_ind[i, 5]
            adc_ind = my_X_all_ind[i, 6]
            h_ind = my_X_all_ind[i, 7]
            q_ind = my_X_all_ind[i, 8]
            # here the points for m are end of t period mortgage leverage at t
            # period prices
            m_ind = my_X_all_ind[i, 9]

            # A_DC_{t+1}, Q_{t+1}, M_{t+1} will be subject to shocks in t+1
            # pull out values of A_DC_{t+1}(1 +R_DC), Q_{t+1} and M_{t+1}(1+r_m)
            # from X_prime_vals

            my_x_prime_vals = gen_x_iid_prime_vals(pi_ind,
                                                   adc_ind, q_ind, m_ind)

            # the m_prime point here is the beggining
            # of t+1 period mortgage leverage at t+1 period price
            point = my_x_prime_vals[:, 0:3]

            R_m_prime = r_m_prime
            ADC_return_prime = my_x_prime_vals[:, 0] / A_DC[adc_ind]

            E_ind_p, alpha_ind_p, beta_ind_p\
                = E_ind, alpha_ind, beta_ind

            # gen values of t+1 unconditioned on t expectation of the iid shocks
            # for DC, Mortgage and house price
            U1, U2, U3, V\
                = eval_linear_c(X_DCQ_W,
                                UC_prime_DCC[DB_ind,
                                             E_ind_p,
                                             alpha_ind_p,
                                             beta_ind_p,
                                             a_ind, :,
                                             h_ind, :, :],
                                point, xto.LINEAR),\
                eval_linear_c(X_DCQ_W,
                              UC_prime_H_DCC[DB_ind,
                                             E_ind_p,
                                             alpha_ind_p,
                                             beta_ind_p,
                                             a_ind, :,
                                             h_ind, :, :],
                              point, xto.LINEAR),\
                eval_linear_c(X_DCQ_W,
                              UC_prime_M_DCC[DB_ind,
                                             E_ind_p, alpha_ind_p,
                                             beta_ind_p, a_ind, :,
                                             h_ind, :, :],
                              point, xto.LINEAR) * R_m_prime,\
                eval_linear_c(X_DCQ_W,
                              VF_DCC[DB_ind, E_ind_p,
                                     alpha_ind_p, beta_ind,
                                     a_ind, :, h_ind, :, :],
                              point, xto.NEAREST)

            my_UC_prime[i], my_UC_prime_H[i],\
                my_UC_prime_M[i], my_VF[i]\
                = np.dot(U1, Q_DC_P),np.dot(U2, Q_DC_P),\
                np.dot(U3, Q_DC_P), np.dot(V, Q_DC_P)
            #print(my_VF[i])

        my_UC_prime[np.where(my_UC_prime <= 0)] = 1E-250
        my_UC_prime_H[np.where(my_UC_prime_H <= 0)] = 1E-250

        my_UC_prime_M[np.where(my_UC_prime_M <= 0)] = 1E-250

        my_UC_prime[np.where(np.isnan(my_UC_prime))] = 1E-250
        my_UC_prime_H[np.where(np.isnan(my_UC_prime_H))] = 1E-250
        my_UC_prime_M[np.where(np.isnan(my_UC_prime_M))] = 1E-250

        return my_UC_prime, my_UC_prime_H, my_UC_prime_M, my_VF

    # Lifecycel model iteration function

    def solve_LC_model(comm, world_comm, load_ret):
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
            ret_path_load = jobfs_path
        else:
            ret_path_load = pol_path_id

        if load_ret == 0:
            # Generate retiree policies
            UC_prime, UC_prime_H, UC_prime_M, VF = gen_R_pol(comm, noplot=True)

            comm.Barrier()
            if comm.rank == 0:
                ret_pols = (UC_prime, UC_prime_H, UC_prime_M, VF)
                pickle.dump(ret_pols, open(
                    "{}/ret_pols_{}.pol" .format(pol_path_id, str(acc_ind[0])), "wb"))
        else:
            pass

        comm.Barrier()
        if load_ret == 1 or comm.rank != 0:
            # Load retiree policies on all ranks
            ret_pols = pickle.load(open("{}/ret_pols_{}.pol"
                                        .format(ret_path_load, str(acc_ind[0])), "rb"))
            (UC_prime, UC_prime_H, UC_prime_M, VF) = ret_pols

        start1 = time.time()

        for Age in np.arange(int(tzero), int(R))[::-1]:
            start2 = time.time()

            # Generate RHS interpolation  points
            points_noadj_vec, points_adj_vec, points_rent_vec, A_prime \
                = gen_points_for_age(Age, comm)

            if verbose and world_comm.rank == 0:
                print(
                    "Generated points for age_{}".format(
                        time.time() - start2))

            if comm.rank != 0:
                points_adj_vec = None
                points_noadj_vec = None
                A_prime = None
                points_rent_vec = None

            start = time.time()
            t = Age

            # Step 1: Evaluate optimal mortgage choice aux. func
            mort_func = None
            if comm.rank == 1:
                mort_func = eval_M_prime(t, UC_prime, UC_prime_M)

            elif comm.rank == 0:

                rent_vals, H_rent = gen_rhs_rent_euler_pols(UC_prime, VF,
                                                            points_rent_vec)
            else:
                pass
            mort_func = comm.bcast(mort_func, root=1)

            # Step 2: Evaluate housing non-adjuster owner policies
            if verbose and world_comm.rank == 0:
                print("Solving for age_{}".format(Age))

            noadjpolsoutparr = gen_rhs_euler_pols_noadj(t, comm,
                                                        mort_func,
                                                        UC_prime,
                                                        UC_prime_H,
                                                        UC_prime_M,
                                                        VF,
                                                        points_noadj_vec)

            # Step 3: Evaluate housing adjuster owner policies
            adjpolsoutparr = gen_rhs_euler_pols_adj(t, comm,
                                                    mort_func,
                                                    UC_prime,
                                                    UC_prime_H,
                                                    UC_prime_M,
                                                    VF,
                                                    points_adj_vec)

            # Label the policies on root
            if comm.rank == 0:

                adj_vals, C_adj, H_adj, Aprime_adj = adjpolsoutparr
                noadj_vals, C_noadj, etas_noadj, Aprime_noadj = noadjpolsoutparr

                if verbose and world_comm.rank == 0:
                    print("Solved eval_policy_W of age {} in {} seconds"
                          .format(Age, time.time() - start))
            else:
                noadj_vals = None
                adj_vals = None
                rent_vals = None

            # Step 4: Evaulate unconditioned RHS of Euler equations
            del points_noadj_vec, points_adj_vec, points_rent_vec
            gc.collect()
            start = time.time()
            B_funcs = gen_rhs_uc_func(comm, t,
                                      VF,
                                      UC_prime_H,
                                      A_prime,
                                      noadj_vals,
                                      adj_vals,
                                      rent_vals)
            UC_prime_DCC = None
            UC_prime_H_DCC = None
            UC_prime_M_DCC = None
            VF_DCC = None

            # Label the RHS of marginal utilities on root
            if comm.rank == 0:
                (UC_prime_B, UC_prime_H_B, UC_prime_M_B,
                 VF_B, Xi, zeta, eta_mu) = B_funcs

                if verbose and world_comm.rank == 0:
                    print("Solved gen_RHS_UC_func of age {} in {} seconds".
                          format(Age, time.time() - start))
            else:
                UC_prime_B = None
                UC_prime_H_B = None
                UC_prime_M_B = None
                VF_B = None
                Xi = None
                zeta = None

            # Step 5: Condition out discrete choice probabilities
            start = time.time()
            UC_prime_DCC, UC_prime_H_DCC,\
                UC_prime_M_DCC, VF_DCC,\
                prob_v, prob_pi = condition_DC(UC_prime_B,
                                               UC_prime_H_B,
                                               UC_prime_M_B, VF_B,
                                               Xi, comm)

            if world_comm.rank == 0:
                if verbose:
                    print("Solved UC_cond_DC of age {} in {} seconds".
                          format(Age, time.time() - start))

            # Step 6: Condition out house price, DC and income shock

            start = time.time()

            UC_prime_DCC = comm.bcast(UC_prime_DCC, root=0)
            UC_prime_H_DCC = comm.bcast(UC_prime_H_DCC, root=0)
            UC_prime_M_DCC = comm.bcast(UC_prime_M_DCC, root=0)
            VF_DCC = comm.bcast(VF_DCC, root=0)

            my_UC_prime, my_UC_prime_H, my_UC_prime_M, my_VF\
                = UC_cond_all(t, UC_prime_DCC,
                              UC_prime_H_DCC,
                              UC_prime_M_DCC,
                              VF_DCC,
                              my_X_all_hat_ind)

            sendcounts = np.array(comm.gather(len(my_UC_prime), 0))

            comm.Gatherv(np.ascontiguousarray(my_UC_prime),
                         recvbuf=(UC_prime, sendcounts), root=0)
            comm.Gatherv(np.ascontiguousarray(my_UC_prime_H),
                         recvbuf=(UC_prime_H, sendcounts), root=0)
            comm.Gatherv(np.ascontiguousarray(my_UC_prime_M),
                         recvbuf=(UC_prime_M, sendcounts), root=0)
            comm.Gatherv(np.ascontiguousarray(my_VF),
                         recvbuf=(VF, sendcounts), root=0)
            del my_UC_prime, my_UC_prime_H, my_UC_prime_M, my_VF

            gc.collect()

            # On all ranks other than 0, reset values of t+1, t conditoned
            # value functions and marginal utilties
            # From rank 0, broadcast value functions and marginal utilties
            # conditioned on time t
            UC_prime = comm.bcast(UC_prime, root=0)
            UC_prime_H = comm.bcast(UC_prime_H, root=0)
            UC_prime_M = comm.bcast(UC_prime_M, root=0)
            VF = comm.bcast(VF, root=0)

            if verbose and world_comm.rank == 0:
                print("Solved UC_cond_all of age {} in {} seconds".
                      format(Age, time.time() - start))
                print("Iteration time was {}".format(time.time() - start2))

            # Step 7: Save policy functions to job file system
            if comm.rank == 0:

                if verbose == True and plot_vf == True:
                    """ Plot value function""" 

                    #import matplotlib.pyplot as plt
                    #import matplotlib.colors as mcolors
                    #import matplotlib.cm as cm
                    #import matplotlib.pyplot as plt
                    #from matplotlib.colors import DivergingNorm


                    NUM_COLORS = grid_size_H
                    colormap = cm.viridis

                    normalize = mcolors.Normalize(
                            vmin=np.min(H), vmax=np.max(H))

                    VF_plot = VF.reshape(all_state_shape)
                    Path('plots/valuefunctions/').mkdir(parents=True, exist_ok=True)

                    for j in range(len(Q)):
                        for h in range(len(H)):
                            plt.plot(np.log(A), VF_plot[0,1,0,1,1,:,3,h,j,1], color=colormap(
                                          h // 3 * 3.0 / NUM_COLORS))

                        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
                        scalarmappaple.set_array(H)
                        cbar = plt.colorbar(scalarmappaple)
                        cbar.set_label('House stock')

                        plt.ylim((-5, 25))
                        plt.xlim((-1, np.log(A[-1])))
                        plt.xlabel('Log Liquid assets (AUD 100,000)')
                        plt.ylabel('Continuation value')
                        plt.savefig('plots/valuefunctions/vf_plot_age_{}_q_{}_acc_{}.png'\
                            .format(t,j, acc_ind[0]))
                        plt.close()


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

                    np.savez_compressed(
                        "{}/age_{}_acc_{}_id_{}_pols". format(
                            pol_path_id,
                            t,
                            acc_ind[0],
                            ID),
                        C_adj=C_adj,
                        H_adj=H_adj,
                        Aprime_adj=Aprime_adj,
                        C_noadj=C_noadj,
                        etas_noadj=np.log(eta_mu).astype(
                            np.float32),
                        Aprime_noadj=Aprime_noadj,
                        zeta=np.log(zeta).astype(
                            np.float32),
                        H_rent=H_rent,
                        prob_v=prob_v.astype(
                            np.float32),
                        prob_pi=prob_pi.astype(
                            np.float32),
                        policy_VF=policy_VF)

                    if verbose and world_comm.rank == 0:
                        print("Saved policies in {} seconds"
                              .format(-time.time() + time.time()))
                    del C_adj, H_adj, Aprime_adj, C_noadj, etas_noadj,\
                        Aprime_noadj, zeta, H_rent, prob_v, prob_pi
                    gc.collect()
                else:
                    np.savez_compressed(
                        "{}/age_{}_acc_{}_id_{}_pols". format(
                            pol_path_id,
                            t,
                            acc_ind[0],
                            ID),
                        C_adj=C_adj,
                        H_adj=H_adj,
                        Aprime_adj=Aprime_adj,
                        C_noadj=C_noadj,
                        etas_noadj=np.log(eta_mu).astype(
                            np.float32),
                        Aprime_noadj=Aprime_noadj,
                        zeta=np.log(zeta).astype(
                            np.float32),
                        H_rent=H_rent,
                        prob_v=prob_v.astype(
                            np.float32),
                        prob_pi=prob_pi.astype(
                            np.float32))

                    if verbose and world_comm.rank == 0:
                        print("Saved policies in {} seconds"
                              .format(start - time.time()))

                    del C_adj, H_adj, Aprime_adj, C_noadj, etas_noadj,\
                        Aprime_noadj, zeta, H_rent, prob_v, prob_pi
                    gc.collect()

        if verbose and world_comm.rank == 0:
            print("Solved lifecycle model in {} seconds".format(
                time.time() - start1))

        del UC_prime, UC_prime_H, UC_prime_M, VF
        gc.collect()
        return ID

    gc.collect()
    return solve_LC_model

def generate_worker_pols(og,
                         world_comm,
                         comm,
                         load_retiree=1,
                         jobfs_path='/scratch/pv33/ls_model_temp2/',
                         verbose=False, plot_vf = False):

    gen_R_pol = retiree_func_factory(og)

    solve_LC_model = worker_solver_factory(og,
                                           world_comm,
                                           comm,
                                           gen_R_pol,
                                           jobfs_path=jobfs_path,
                                           verbose=verbose, plot_vf = plot_vf)

    policies = solve_LC_model(comm, world_comm, load_retiree)

    del solve_LC_model, gen_R_pol, og
    gc.collect()
    return policies
