  
"""
Module contains the HousingModel retiree 
solvers for Bateman et al (2020)
 
Functions: housing_model_retiree_func_factory
            Generates the operators require to solve retiree 
            policy functions from an instance \
            of Housing Model 

Akshay Shanker
School of Economics 
University of New South Wales
akshay.shanker@me.com

"""

import numpy as np
from numba import njit, prange, vectorize, int64, float64,guvectorize
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interpolation import interp
from quantecon.optimize.root_finding import brentq, newton_secant
from interpolation.splines import extrap_options as xto
from time import time

def retiree_func_factory(og):
    """A function that creates operator 
        to solve retiree problem using NC-EGM  

    Parameters*2
    ----------
    og :        HousingModel 
                 Instance of housing model class

    Returns
    -------
    gen_R_pol: function 
                generates policy for retiring workers
                at age 64 

    """

    # Functions
    u, uc, uh,uc_inv, uh_inv        = og.functions.u, og.functions.uc, og.functions.uh, og.functions.uc_inv,\
                                        og.functions.uh_inv
    b, b_prime                      = og.functions.b, og.functions.b_prime 
    y, DB_benefit                   = og.functions.y, og.functions.DB_benefit
    adj_p, adj_v,adj_pi             = og.functions.adj_p, og.functions.adj_v, og.functions.adj_pi
    amort_rate                      = og.functions.amort_rate
   
    # Parameters
    k                               = og.parameters.k
    phi_r                           = og.parameters.phi_r
    A_max_WE                        = og.parameters.A_max_WE
    delta_housing, alpha, beta_bar  = og.parameters.delta_housing, og.parameters.alpha,\
                                        og.parameters.beta_bar
    tau_housing                     = og.parameters.tau_housing    
    r,s, r_H                        = og.parameters.r, og.parameters.s, og.parameters.r_H     
    r_l, beta_m, kappa_m            = og.parameters.r_l, og.parameters.beta_m, og.parameters.kappa_m
    alpha_housing                   = og.st_grid.alpha_housing

    Q_shocks_r, Q_shocks_P          = og.st_grid.Q_shocks_r, og.st_grid.Q_shocks_P
    Q_DC_shocks, Q_DC_P             = og.cart_grids.Q_DC_shocks, og.cart_grids.Q_DC_P 
    X_QH_R                          = og.interp_grid.X_QH_R 

    H_R, HR_Q                       = og.grid1d.H_R, og.cart_grids.HR_Q             
    A,A_DC, Q, H, M, W_R            = og.grid1d.A,og.grid1d.A_DC, og.grid1d.Q, og.grid1d.H, og.grid1d.M,og.grid1d.W_R
    A_R, H_Q, A_Q_R,W_Q_R           = og.grid1d.A_R, og.cart_grids.H_Q, og.cart_grids.A_Q_R,og.interp_grid.W_Q_R
    E, P_E, P_stat                  = og.st_grid.E, og.st_grid.P_E, og.st_grid.P_stat

    A_min, C_min, C_max, \
    H_min, A_max_R                  = og.parameters.A_min, og.parameters.C_min, og.parameters.C_max,\
                                         og.parameters.H_min, og.parameters.A_max_R
    H_max                           = og.parameters.H_max

    X_all_hat_ind                   = og.big_grids.X_all_hat_ind
    X_all_hat_vals                  = og.big_grids.X_all_hat_vals

    X_cont_R,X_R_contgp,\
    X_H_R_ind,\
    X_RC_contgp, X_R_cont_ind      = og.interp_grid.X_cont_R, og.interp_grid.X_R_contgp,\
                                        og.cart_grids.X_H_R_ind,\
                                        og.interp_grid.X_RC_contgp,\
                                        og.cart_grids.X_R_cont_ind

    grid_size_A, grid_size_DC,\
    grid_size_H, grid_size_Q,\
    grid_size_M, grid_size_C        = og.parameters.grid_size_A,\
                                        og.parameters.grid_size_DC,\
                                        og.parameters.grid_size_H,\
                                        og.parameters.grid_size_Q,\
                                        og.parameters.grid_size_M,\
                                        og.parameters.grid_size_C
    grid_size_HS                     = og.parameters.grid_size_HS

    T, tzero, R                     = og.parameters.T, og.parameters.tzero, og.parameters.R

    @njit
    def interp_as(xp,yp,x):
        """ interpolates 1D
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
        return evals

    @njit
    def interp_adj(a_adj,c_adj, wealth_endgrid, extrap = True):
        """Reshapes and interpolates policy functions
            for housing adjusters on endogenous wealth 
            grid """

        a_adj_uniform       = np.zeros((grid_size_Q*grid_size_M, grid_size_A))
        H_adj_uniform       = np.zeros((grid_size_Q*grid_size_M, grid_size_A))
        c_adj_uniform       = np.zeros((grid_size_Q*grid_size_M, grid_size_A))


        a_adj_bar           = np.transpose(a_adj.reshape(grid_size_H, \
                                    grid_size_Q*grid_size_M))
        wealth_endgrid_bar  = np.transpose(wealth_endgrid.reshape(grid_size_H, \
                                    grid_size_Q*grid_size_M))
        c_adj_bar           = np.transpose(c_adj.reshape(grid_size_H, \
                                    grid_size_Q*grid_size_M))


        for i in prange(len(wealth_endgrid_bar)):

            wealthbar_c = wealth_endgrid_bar[i]
            A_bar_c     = a_adj_bar[i]
            H_c         = H
            C_c         = c_adj_bar[i]
            wealth_x    =  wealthbar_c[~np.isnan(wealthbar_c)]

            assts_x     = np.take(A_bar_c[~np.isnan(wealthbar_c)],\
                                     np.argsort(wealth_x))
            cons_x      = np.take(C_c[~np.isnan(wealthbar_c)],\
                                     np.argsort(wealth_x))

            
            h_x         = np.take(H_c[~np.isnan(wealthbar_c)],\
                                     np.argsort(wealth_x))

            #sprint(h_x)

            wealth_x_sorted             = np.sort(np.copy(wealth_x))
            h_x[wealth_x_sorted<=A_min] = H_min
            #print(wealth_x_sorted)

            if extrap == True:
                c_adj_uniform[i]        = interp_as(wealth_x_sorted,cons_x,W_R)

                c_adj_uniform[i][ c_adj_uniform[i] <=C_min] =  C_min
                c_adj_uniform[i][ c_adj_uniform[i] > C_max]= C_max

                a_adj_uniform[i]        = interp_as(wealth_x_sorted,assts_x, W_R)

                a_adj_uniform[i][a_adj_uniform[i]<=A_min] = A_min

                H_adj_uniform[i]        = interp_as(wealth_x_sorted,h_x,W_R)
                H_adj_uniform[i][ H_adj_uniform[i] <=H_min]   = H_min
            else:
                c_adj_uniform[i]        = np.interp(W_R,wealth_x_sorted,cons_x)

                c_adj_uniform[i][ c_adj_uniform[i] <=C_min] =  C_min
                c_adj_uniform[i][ c_adj_uniform[i] > C_max]= C_max

                a_adj_uniform[i]        = np.interp(W_R,wealth_x_sorted,assts_x)

                a_adj_uniform[i][a_adj_uniform[i]<=A_min] = A_min

                H_adj_uniform[i]        = np.interp(W_R,wealth_x_sorted,h_x)
                H_adj_uniform[i][ H_adj_uniform[i] <=H_min]   = H_min



            H_adj_uniform[i][0] = H_min

        return np.reshape(a_adj_uniform,(grid_size_Q,grid_size_M, grid_size_A)),\
                np.reshape(c_adj_uniform,(grid_size_Q,grid_size_M, grid_size_A)),\
                np.reshape(H_adj_uniform,(grid_size_Q,grid_size_M, grid_size_A))
    @njit
    def interp_no_adj(assets_endgrid_1,cons_1,etas_1):

        """ Reshapes and interps the policy functions
            for housing non-adjusters on endogenous a
            assett grid""" 

        assets_endgrid              = assets_endgrid_1.reshape(grid_size_A,\
                                                    grid_size_H\
                                                    *grid_size_Q*grid_size_M)
        assets_endgrid              = np.transpose(assets_endgrid)

        cons_reshaped               = cons_1.reshape(grid_size_A,\
                                                    grid_size_H\
                                                    *grid_size_Q*grid_size_M)
        cons_reshaped               = np.transpose(cons_reshaped)

        etas_reshaped               = etas_1.reshape(grid_size_A,\
                                                    grid_size_H\
                                                    *grid_size_Q*grid_size_M)
        etas_reshaped               = np.transpose(etas_reshaped)



        assets_uniform              = np.zeros((grid_size_H*grid_size_Q*grid_size_M,grid_size_A) )
        etas_uniform                = np.zeros((grid_size_H*grid_size_Q*grid_size_M,grid_size_A) )
        cons_uniform                = np.zeros((grid_size_H*grid_size_Q*grid_size_M,grid_size_A) )


        for i in prange(len(assets_uniform)):

            # interp_as next period assets on current period 
            # endogenous grid of assets 
            assets_uniform[i] = interp_as(np.sort(assets_endgrid[i]),\
                                                np.take(A_R,\
                                                np.argsort(assets_endgrid[i])),A_R)


            assets_uniform[i][assets_uniform[i]<0] = A_min


            # interp_as etas on current period 
            # endogenous grid of assets 
            
            etas_uniform[i]   = interp_as(np.sort(assets_endgrid[i]),\
                                                np.take(etas_reshaped[i],\
                                                np.argsort(assets_endgrid[i])),A_R)

            # interp_as consumption at t on current period 
            # endogenous grid of assets 
            cons_uniform[i]   = interp_as(np.sort(assets_endgrid[i]),\
                                                np.take(cons_reshaped[i],\
                                                np.argsort(assets_endgrid[i])),A_R)

            #print(cons_reshaped[i])

            cons_uniform[i][cons_uniform[i] <0] = C_min


        #print (A_R- assets_uniform[i]-cons_uniform[i])

        # re-shape interpolated policies on time t state
        
        a_noadj_1              =  np.transpose(assets_uniform)
        a_noadj                =  np.reshape(np.ravel(a_noadj_1),\
                                                (grid_size_A,\
                                                grid_size_H,\
                                                grid_size_Q,\
                                                grid_size_M))

        etas_noadj_1         =  np.transpose(etas_uniform)
        etas_noadj           =  np.reshape(np.ravel(etas_noadj_1),\
                                                (grid_size_A,\
                                                grid_size_H,\
                                                grid_size_Q,\
                                                grid_size_M))

        c_noadj_1                =  np.transpose(cons_uniform)
        c_noadj                  =  np.reshape(np.ravel(c_noadj_1),\
                                                (grid_size_A,\
                                                grid_size_H,\
                                                grid_size_Q,\
                                                grid_size_M))



        return a_noadj, c_noadj, etas_noadj


    @njit 
    def rent_FOC(c,s,q):    

        RHS   = uh(c,s, alpha_housing)/(q*phi_r)

        return c - uc_inv(RHS, s, alpha_housing)

    @njit
    def gen_rent_pol():

        cons = np.zeros(len(H_Q))

        for i in prange(len(H_Q)):
            cons[i] = brentq(rent_FOC, 1e-100, 100, args = (H_Q[i,0], H_Q[i,1]))[0]

        #cons_out = cons.reshape(len(H),len(Q) )

        return cons 

    #cons_rent = gen_rent_pol()

    @njit
    def liq_rent_FOC(a_prime,cons,h,q, t_prime_funcs,t):


        UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
        UC_prime_M_RHS, UF\
                            = gen_UC_RHS(t,a_prime,H_min,q,0,\
                                     *t_prime_funcs)
        #print(UC_prime_RHS)

        RHS = UC_prime_RHS

        return uc(cons,h, alpha_housing) - RHS



    @njit
    def eval_rent_pol(t_prime_funcs,t):

        a_end_1         = np.zeros(len(HR_Q))
        #a_prime_1       = np.zeros(len(HR_Q))
        #cons_end_1      = np.zeros(len(HR_Q))

        for i in prange(len(HR_Q)):
            #c_t             = cons_rent[i]

            c_t             = phi_r*HR_Q[i,1]*HR_Q[i,0]*(1-alpha_housing)/alpha_housing

            #cons_end_1[i]      = c_t

            rent_focargs    = (c_t, HR_Q[i,0], HR_Q[i,1], t_prime_funcs,t)

            if liq_rent_FOC(A_min, *rent_focargs)*liq_rent_FOC(A_max_WE, *rent_focargs)<0:
                a_prime_1     = brentq(liq_rent_FOC, A_min,A_max_WE, args = rent_focargs )[0]
            elif liq_rent_FOC(A_min, *rent_focargs)>0:
                a_prime_1    = A_min
            elif liq_rent_FOC(A_max_WE, *rent_focargs)<0:
                a_prime_1    = A_max_WE
            else:
                a_prime_1       = np.nan
                a_end_1[i]      = np.nan

            a_end_1[i]          = c_t + a_prime_1 +  HR_Q[i,1]*phi_r*HR_Q[i,0]

        a_end       = np.transpose(a_end_1.reshape((int(grid_size_HS),len(Q) )))

        h_prime_func = np.zeros((len(Q), len(A)))

        for i in range(len(Q)):

            #print(a_end[i])

            h_prime_func[i,:] = interp_as(np.sort(a_end[i][a_end[i]!=np.nan]),np.take(H_R[a_end[i]!=np.nan],np.argsort(a_end[i][a_end[i]!=np.nan])) , W_R)

            h_prime_func[i,:][h_prime_func[i,:]<=0] = H_min

        return   np.transpose(h_prime_func)


    @njit 
    def HA_FOC(x_prime,\
                h,\
                q,\
                m,\
                m_prime_func,\
                t_prime_funcs,\
                t,\
                ret_cons = False,\
                ret_mort = False):
       
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
                             time t mortgage liability
        mort_func:          4D array 
                             time t+1 mortgage if unconstrainted 
                             adjustment function of 
                             a_t+1, h_t, q_t, c_t
        t_prime_funcs:      6-tuple
                             next period policy functions
       
        t:                  int
                             Age
        Returns
        -------
        
        Euler error:        float64

        """

        m_prime, c_t        = eval_c_mort(x_prime, h,q,m,\
                                          m_prime_func,\
                                          t_prime_funcs,t)

        UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
        UC_prime_M_RHS, UF\
                            = gen_UC_RHS(t,x_prime,h,q,m_prime,\
                                     *t_prime_funcs)

        c_t                 = max(C_min,uc_inv(UC_prime_RHS, h, alpha_housing))

        RHS                 = uc(c_t, h, alpha_housing)*q*(1+tau_housing)\
                                 - UC_prime_H_RHS

        # return equation x in paper 

        if ret_cons == True:
            return c_t

        elif ret_mort == True:
            return m_prime

        elif UC_prime_HFC_RHS!=0:
            return np.abs((uh(c_t,h,alpha_housing) - RHS)) \
                            - UC_prime_HFC_RHS
        else:

            return uh(c_t,h,alpha_housing) - RHS

    @njit 
    def H_FOC(c, x_prime,\
                h,\
                q,\
                m,\
                t_prime_funcs,t,\
                ret_mort = False):
       
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

        UC_prime_RHSf, UC_prime_H_RHSf, UC_prime_HFC_RHSf,\
        UC_prime_M_RHSf, UFf\
                        = gen_UC_RHS(t,x_prime,h,q,0,\
                                     *t_prime_funcs)

        if UC_prime_RHSf<  UC_prime_M_RHSf:
            m_prime = 0

        else:
            m_prime     = m - amort_rate(t-2)*m

        UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
        UC_prime_M_RHS, UF\
                = gen_UC_RHS(t,x_prime,h,q,m_prime,\
                             *t_prime_funcs)

        RHS             = uc(c, h, alpha_housing)*q*(1+tau_housing)\
                            - UC_prime_H_RHS

        if ret_mort == True:
            return m_prime 

        else:
            return c - uc_inv(RHS,h, alpha_housing)

    @njit
    def mort_FOC(m, x_prime,\
                h,\
                q,\
                t_prime_funcs,t):

        """ FOC for interior mortage decision
            i.e. unconstrained by min amort. 
            payment

            Equation x in paper"""

        UC_prime_RHS, UC_prime_H_RHS,\
        UC_prime_HFC_RHS, UC_prime_M_RHS, UF\
                    = gen_UC_RHS(t,x_prime,h,\
                                 q,m, *t_prime_funcs)

        return UC_prime_RHS- UC_prime_M_RHS

    @njit 
    def eval_c_mort(x_prime,\
                    h,\
                    q,\
                    m,\
                    m_prime_func,\
                    t_prime_funcs,t):

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
                             time t mortgage liability (after interest)
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

        m_prime_m         = (1-amort_rate(t-2))*m

        UC_prime_RHSm, UC_prime_H_RHSm, UC_prime_HFC_RHSm,\
        UC_prime_M_RHSm, UFf\
                = gen_UC_RHS(t,x_prime,h,q,m_prime_m,\
                             *t_prime_funcs)

        UC_prime_RHSf, UC_prime_H_RHSf, UC_prime_HFC_RHSf,\
        UC_prime_M_RHSf, UFf\
                = gen_UC_RHS(t,x_prime,h,q,0,\
                             *t_prime_funcs)

        m_mort_args = (x_prime, h, q,t_prime_funcs, t)

        if UC_prime_RHSm> UC_prime_M_RHSm:
            m_prime         = m*(1-amort_rate(t-2))

            UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
            UC_prime_M_RHS, UF\
                = gen_UC_RHS(t,x_prime,h,q,m_prime,\
                             *t_prime_funcs)

            c_t             = uc_inv(max(1e-200,UC_prime_RHS),\
                                        h,\
                                        alpha_housing)

        elif UC_prime_RHSf< UC_prime_M_RHSf:
            m_prime = 0 

            UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
            UC_prime_M_RHS, UF\
                = gen_UC_RHS(t,x_prime,h,q,m_prime,\
                             *t_prime_funcs)

            c_t             = uc_inv(max(1e-200,UC_prime_RHS),\
                                        h,\
                                        alpha_housing)
        else:
            m_prime = max(0,min(m_prime_m,interp(A_R,m_prime_func,\
                                        x_prime)))


            UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
            UC_prime_M_RHS,UF\
                = gen_UC_RHS(t,x_prime,h,q,m_prime,\
                             *t_prime_funcs)

            c_t             = max(C_min, uc_inv(max(1e-200,UC_prime_RHS),\
                                        h, alpha_housing))

        return c_t, m_prime 

    @njit 
    def gen_UC_RHS(t,x_prime,\
                    h,\
                    q,
                    m,\
                    a_prime_noadj,\
                    c_prime_noadj,\
                    eta_prime_noadj,\
                    a_prime_adj,\
                    c_prime_adj,\
                    h_prime_adj,\
                    zeta_nl,\
                    c_prime_adj_nl,\
                    h_prime_adj_nl,\
                    h_prime_rent, \
                    UF_dbprime):
        
        """ At time t, Evaluates RHS value of Euler equation for t+1 
                    Equation x in paper

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
                         m_t+1 mortgage liability 
                        (before t+1 interest!)
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

        Returns
        -------
        UC_prime_RHS:       float64
                             RHS of equation x
        UC_prime_H_RHS:     float64
                             RHS of equation x
        UC_prime_HFC_RHS:   float64
                             RHS of equation x
        UC_prime_M_RHS:     float64
                             RHS of equation x

        
        Note: t+1 A is assets before returns, H is housing after dep. 
        and W is wealth in hand after returns and sale of house next 
        with period rices

        Note2: check intuition for why mort_prime_func plays no role
                in this function/ 
                what happens to the 

        """

        #STEP1: Evaluate t+ 1 states 

        # array of t+1 house prices, mort. interest rates,
        # mortgage balnces after interest and net wealth 

        Q_prime,r_m_prime      = q*(1+r_H + Q_DC_shocks[:,2]),\
                                beta_m*r_l*(Q_DC_shocks[:,0]/r_l)**kappa_m
                                            


        M_prime                       = (1+r_m_prime)*m

        W_prime  = (1+r)*x_prime - amort_rate(t+1-2)*M_prime \
                                        + Q_prime*np.full(len(Q_prime),\
                                                    (1-delta_housing)*h)
        
        # t+1 states: A_t+1(1+r) - min mort payment payment, P_t+1, M_t+1 (1+r_m)
        # this is the state for the no  housing adjusters

        state_prime_R                   =  np.column_stack(((x_prime*(1+r)\
                                            - amort_rate(t+1-2)*M_prime), \
                                            np.full(len(Q_prime),\
                                            (1-delta_housing)*h),\
                                            Q_prime, M_prime))

        state_prime_R[:,0][state_prime_R[:,0]<=0] = A_min

        # t+1 states: P_t+1, M_t+1 (1+r_m), net wealth - min mort payment
        # this is the state for the   housing adjusters

        state_prime_RW          =  np.column_stack((Q_prime,M_prime,\
                                                    W_prime))

        # t+1 states: t+1 states: P_t+1, M_t+1 (1+r_m),
        # net wealth - min mort payment - housing adjustment cost 
        # this is state for renters 

        state_prime_rent        =  np.column_stack((Q_prime,W_prime \
                                                - M_prime*(1-amort_rate(t+1-2)\
                                                - tau_housing*Q_prime*h)))

        cannot_rent_1           = state_prime_rent[:,1]<0

        #cannot_rent             = np.sum(cannot_rent_1)>0

        state_prime_rent[:,1][state_prime_rent[:,1]<0]   = A_min

        # bequest value 

        A_prime                 = max(x_prime*(1+r) \
                                    + (1-delta_housing)*h*q -m,A_min)           # should there be a (1+r) here that goes into the bequest function?
                                                                                # the bequest function should have the *Next* period house price and 
                                                                                # mortgage repayment rate 
        # STEP 2: evaluate multipliers 
        #      eta_ind> 1 if NOT adjusting housing stock (cond on not renting)
        #      zeta_ind>1 if NOT making liquid saving (cond on not adjusting)

        # evaluate array of next period eta adjustment multipliers 

        eta_primes_vals         = eval_linear(X_cont_R, eta_prime_noadj,\
                                     state_prime_R,xto.LINEAR)

        # evaluate where adjustment occurs 
        # adjustment occurs where someone
        # defaults on mort. payment
        # or has eta >1 

        eta_ind1                = np.abs(eta_primes_vals)<=1 

        nodefault               = state_prime_R[:,0]>=0  

        eta_ind                 = (eta_ind1>=1)

        # evalute zetas 

        zeta_prime_adj_vals_nl      = eval_linear(X_QH_R,\
                                        zeta_nl,\
                                        state_prime_RW,\
                                        xto.LINEAR)

        zeta_ind                    = zeta_prime_adj_vals_nl>1


        #zeta_ind = np.zeros(len(Q_DC_P))
        # STEP 3: calc cons and a_prime for 
        # adjusters if liq. saving made

        c_prime_adj_vals, a_prime_adj_vals,\
        h_prime_adj_vals             = eval_linear(X_QH_R,\
                                                c_prime_adj,\
                                                state_prime_RW,\
                                                xto.LINEAR),\
                                     eval_linear(X_QH_R,\
                                                a_prime_adj,\
                                                state_prime_RW,\
                                                xto.LINEAR), \
                                     eval_linear(X_QH_R,\
                                                h_prime_adj,\
                                                state_prime_RW,\
                                                xto.LINEAR)

        c_prime_adj_vals[state_prime_RW[:,2]\
                                <=0] =  C_min

        c_prime_adj_vals[c_prime_adj_vals<C_min]  = C_min

        h_prime_adj_vals[h_prime_adj_vals<H_min]  = H_min

        h_prime_adj_vals[state_prime_RW[:,2]<=0]  = H_min

        # STEP 4: calc cons and a_prime for 
        # non-adjusters 
        
        c_prime_noadj_vals,a_prime_noadj_vals  = eval_linear(X_cont_R,\
                                                        c_prime_noadj,\
                                                        state_prime_R,\
                                                        xto.LINEAR),\
                                                    eval_linear(X_cont_R,\
                                                        a_prime_noadj,\
                                                        state_prime_R,\
                                                        xto.LINEAR)

        c_prime_noadj_vals[c_prime_noadj_vals<C_min] = C_min
        c_prime_noadj_vals[state_prime_R[:,0]<=0]    = C_min

        
        h_prime_noadj_vals              = np.full(len(Q_prime),\
                                                (1-delta_housing)*h)
        h_prime_noadj_vals[h_prime_noadj_vals<0] = H_min

        # STEP 5: calc cons and a_prime for 
        # adjusters wiht no liq saving 

        h_prime_adj_vals_nl,c_prime_adj_vals_nl     = eval_linear(X_QH_R,\
                                                        h_prime_adj_nl,\
                                                        state_prime_RW,\
                                                        xto.LINEAR),\
                                                     eval_linear(X_QH_R,\
                                                        c_prime_adj_nl,\
                                                        state_prime_RW,\
                                                        xto.LINEAR)

        h_prime_adj_vals_nl[h_prime_adj_vals_nl<H_min] = H_min
        c_prime_adj_vals_nl[c_prime_adj_vals_nl<C_min] = C_min


        # STEP 6: calculate mortgage payment and flag extra payment

        mort_expay_noadj        = state_prime_R[:,0] - c_prime_noadj_vals\
                                                     - a_prime_noadj_vals

        mort_expay_adj          = state_prime_RW[:,2] - c_prime_adj_vals\
                                                      - a_prime_adj_vals\
                                                      - h_prime_adj_vals\
                                                        *Q_prime*(1+tau_housing)

        mort_expay_adj_nl       = state_prime_RW[:,2] - c_prime_adj_vals_nl\
                                                      - h_prime_adj_vals_nl\
                                                        *Q_prime*(1+tau_housing)


        mort_ex_pay             = mort_expay_noadj*eta_ind\
                                    + ((1-zeta_ind)*mort_expay_adj\
                                    + zeta_ind*mort_expay_adj_nl)*(1-eta_ind)
        
        # STEP 7: create vec of t+2 mortgage balances (before t+2 interest )

        mort_dp_prime           = (1-amort_rate(t-2+1))*m- mort_ex_pay

        mort_dp_prime[mort_dp_prime<0] = 0

        extra_pay_norent               = mort_ex_pay>1e-5

        # STEP 8: combine all non-renting policies 
        c_prime_val_norent                     = ((1-zeta_ind)*c_prime_adj_vals\
                                            + zeta_ind*c_prime_adj_vals_nl)\
                                            *(1-eta_ind)\
                                            + c_prime_noadj_vals*eta_ind

        c_prime_val_norent[c_prime_val_norent <C_min]  = C_min

        h_prime_val_norent                      = ((1-zeta_ind)*h_prime_adj_vals \
                                                    + zeta_ind*h_prime_adj_vals_nl)*(1-eta_ind)\
                                                    + h_prime_noadj_vals*eta_ind  

        h_prime_val_norent[h_prime_val_norent <H_min]  = H_min

        a_prime_val_norent                     = (1-zeta_ind)*a_prime_adj_vals*(1-eta_ind)\
                                            + a_prime_noadj_vals*eta_ind  

        a_prime_val_norent[a_prime_val_norent<=A_min] = A_min

        #   t+2 states if not renting and discounred t+2 utility (on t+1 information)
        state_dp_prime_norent      = np.column_stack((a_prime_val_norent, h_prime_val_norent,\
                                                      Q_prime, mort_dp_prime))

        UF_dp_val_norent            = beta_bar*eval_linear(X_cont_R, UF_dbprime,\
                                                     state_dp_prime_norent )

        #   t+1 marginal utility of consumption for non-retning 

        uc_prime_norent                = uc(c_prime_val_norent,\
                                     h_prime_val_norent, alpha_housing)
        # STEP 9: combine all  renter policies 

        h_prime_rent_val        = eval_linear(W_Q_R,h_prime_rent,\
                                                state_prime_rent, xto.LINEAR)

        c_prime_rent_val        = phi_r*Q_prime*h_prime_rent_val\
                                        *(1-alpha_housing)/alpha_housing

        c_prime_rent_val[c_prime_rent_val<=C_min] = C_min
        h_prime_rent_val[h_prime_rent_val<=H_min] = H_min

        a_prime_rent_val        = state_prime_rent[:,1] - c_prime_rent_val\
                                     - h_prime_rent_val*phi_r*Q_prime

        state_dp_prime_rent     = np.column_stack((a_prime_rent_val,\
                                        np.full(len(a_prime_rent_val), H_min),\
                                        Q_prime, np.full(len(a_prime_rent_val), 0)))

        #  t+1 marginal utility with renting 
        uc_prime_rent           = uc(c_prime_rent_val,\
                                        h_prime_rent_val, alpha_housing)

        u_prime_rent           = u(c_prime_rent_val,\
                                    h_prime_rent_val, alpha_housing)

        u_prime_norent           = u(c_prime_val_norent,\
                                        h_prime_val_norent, alpha_housing)

        UF_dp_val_rent           = beta_bar*eval_linear(X_cont_R, UF_dbprime,\
                                             state_dp_prime_rent)

        # STEP 10: make renting vs. no renting decision  and combine all policies 

        renter                  = (u_prime_rent + UF_dp_val_rent >u_prime_norent +UF_dp_val_norent)\
                                    *(1-cannot_rent_1)

        #renter = np.zeros(len(Q_DC_P))

        #print(renter)

        h_prime_val             = renter*h_prime_rent_val + (1- renter)*h_prime_val_norent
        c_prime_val             = renter*c_prime_rent_val + (1- renter)*c_prime_val_norent
        extra_pay               = renter*1            + (1-renter)*extra_pay_norent
        uc_prime                = renter*uc_prime_rent + (1-renter)*uc_prime_norent
        
        
        # STEP 11: t+1 utilities conditioned in t info (renter) 

        UC_prime                = np.dot(s[t]*uc_prime*(1+r) +\
                                        (1-s[t])*(1+r)*b_prime(A_prime),
                                        Q_DC_P)

        UC_prime_H              = np.dot(((1-delta_housing- tau_housing*renter)*Q_prime)*(s[t]*uc_prime
                                        +   (1-s[t])*b_prime(A_prime)) ,Q_DC_P)

        UC_prime_HFC            = np.dot(s[t]*uc_prime\
                                        *(tau_housing*Q_prime*h_prime_val)\
                                        *eta_ind*(1-renter),Q_DC_P)

        UC_prime_M_inner        = (1+r_m_prime)*(extra_pay)\
                                    *s[t]*uc_prime + (1-s[t])*(1+r_m_prime)\
                                                    *b_prime(A_prime)

        UC_prime_M              = np.dot(UC_prime_M_inner,Q_DC_P)

        UF_inner                = u(c_prime_val,\
                                        h_prime_val, alpha_housing) 

        UF                      = np.dot(s[t]*UF_inner +\
                                        (1-s[t])*b(A_prime),
                                        Q_DC_P)
       
        # discount everything back 

        UC_prime_RHS            = beta_bar*UC_prime

        UC_prime_H_RHS          = beta_bar*UC_prime_H
   
        UC_prime_HFC_RHS        = beta_bar*UC_prime_HFC

        UC_prime_M_RHS          = beta_bar*UC_prime_M


        return UC_prime_RHS, UC_prime_H_RHS,UC_prime_HFC_RHS,\
                 UC_prime_M_RHS, UF


    @njit
    def eval_mort_policy(t, t_prime_funcs):
        """ returns unconstrained next period mortgage m_t+1 as function 
            of a_t+1, h_t and q_t"""

        m_prime_func        =   np.empty(grid_size_A*grid_size_H\
                                        *grid_size_Q)

        # loop over values of A_t+1, H_t, Q_t
        for i in range(len(X_RC_contgp)):

            # pull out state values for i 
            x_prime,h,q     = X_RC_contgp[i][0],X_RC_contgp[i][1],\
                                X_RC_contgp[i][2]
            
            m_mort_args     = (x_prime, h, q,t_prime_funcs, t)

            m_prime_m       = M[-1]

            # get RHS of Euler equation when max. mortgage taken 
            # (max given by grid max)
            UC_prime_RHSm, UC_prime_H_RHSm, UC_prime_HFC_RHSm,\
            UC_prime_M_RHSm, UFm\
                            = gen_UC_RHS(t,x_prime,h,q,m_prime_m,\
                                 *t_prime_funcs)

            # get RHS of Euler when min mortage taken
            # (no negative mortgages )
            UC_prime_RHSf, UC_prime_H_RHSf, UC_prime_HFC_RHSf,\
            UC_prime_M_RHSf, UFf\
                            = gen_UC_RHS(t,x_prime,h,q,0,\
                                 *t_prime_funcs)

            # check if m_t+1 is constrained by max mortgage
            if UC_prime_RHSm>= UC_prime_M_RHSm:
                m_prime_func[i]         = m_prime_m

            # check if m_t+1 is constrained by min mortgage
            elif UC_prime_RHSf<=UC_prime_M_RHSf:
                m_prime_func[i]         = 0 

            # otherwise, solve for interior unconstrained mortgage
            else:
                m_prime_func[i]         = brentq(mort_FOC, 0,M[-1],\
                                             args= m_mort_args)[0]

        # reshape to wide and return function 
        return m_prime_func.reshape(grid_size_A,grid_size_H,grid_size_Q)


    @njit
    def eval_policy_R_noadj(t,m_prime_func, t_prime_funcs):
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

        assets_endgrid_1   =   np.empty(grid_size_A*grid_size_H\
                                        *grid_size_Q*grid_size_M)
        cons_1             =   np.empty(grid_size_A*grid_size_H\
                                        *grid_size_Q*grid_size_M)
        etas_1             =   np.empty(grid_size_A*grid_size_H\
                                        *grid_size_Q*grid_size_M)

        # loop over values of A_t+1, H_t, Q_t, M_t
        for i in  prange(len(X_R_contgp)):

            x_cont_vals             = X_R_contgp[i] 

            ap_ind, h_ind, q_ind    = X_R_cont_ind[i][0],\
                                        X_R_cont_ind[i][1],\
                                        X_R_cont_ind[i][2]

            # return optimal next period mortgage value and 
            # t period consumption, asssign consumption to grid 

            cons_1[i], m_prime      = eval_c_mort(x_cont_vals[0],\
                                                x_cont_vals[1],\
                                                x_cont_vals[2],\
                                                x_cont_vals[3],\
                                                m_prime_func[:,h_ind,\
                                                q_ind],\
                                                t_prime_funcs,t)

            # calculate extra mortgage payment i.e. pay above min. amort
            extra_payment_made      = (1-amort_rate(t-2))*x_cont_vals[3]\
                                            - m_prime
            # assign A_t value to endogenous grid 

            assets_endgrid_1[i]     = cons_1[i] + x_cont_vals[0]\
                                                + extra_payment_made

            # eval. RHS values of Euler at optimum 
            UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
            UC_prime_M_RHS,UF\
                                    = gen_UC_RHS(t,x_cont_vals[0],\
                                            x_cont_vals[1],\
                                            x_cont_vals[2],\
                                            m_prime,\
                                            *t_prime_funcs)
            
            # calculate adjustment eta multipler for grid
            # point, see equation x

            geta_t  = (uh(max(1e-140,cons_1[i]),x_cont_vals[1],\
                        alpha_housing) +UC_prime_H_RHS \
                        - uc(cons_1[i],x_cont_vals[1],alpha_housing)\
                            *x_cont_vals[2])


            etas_1[i] = geta_t/(uc(max(1e-140,cons_1[i]),x_cont_vals[1],\
                        alpha_housing)\
                        *x_cont_vals[2]\
                        *x_cont_vals[1]\
                        *tau_housing\
                        + UC_prime_HFC_RHS)

        # interpolate A_t+1, C_t and eta_t on endogenous 
        # grid points, i.e on time t states 

        a_noadj, c_noadj, etas_noadj\
                         = interp_no_adj(assets_endgrid_1,cons_1,etas_1)

        return a_noadj, c_noadj, etas_noadj


    @njit
    def eval_policy_R_adj(t,m_prime_func, t_prime_funcs):
        
        """ Generate policy functions with housing stcok adjustment
                and non-zero liquid saving A_t+1
        """

        a_adj           = np.zeros(grid_size_H*grid_size_Q*grid_size_M)
        wealth_endgrid  = np.zeros(grid_size_H*grid_size_Q*grid_size_M)
        c_adj           = np.zeros(grid_size_H*grid_size_Q*grid_size_M)


        for i in range(len(X_H_R_ind)):
            h_index = X_H_R_ind[i][0]
            q_index = X_H_R_ind[i][1]
            m_index = X_H_R_ind[i][2]

            args_HA_FOC = (H[h_index],\
                            Q[q_index],\
                            M[m_index],\
                            m_prime_func[:,h_index,q_index],\
                            t_prime_funcs,t)

                # check if interior solution for a_t+1 exists
            if HA_FOC(A_min, *args_HA_FOC )*HA_FOC(A_max_R, *args_HA_FOC)<0:

                 # if interior solution to a_t+1, calculate it 
                a_adj[i]        = max(brentq(HA_FOC, A_min,A_max_R,\
                                    args = args_HA_FOC)[0], A_min)


                
                c_adj[i]        = max(HA_FOC(a_adj[i],H[h_index],\
                                        Q[q_index],\
                                        M[m_index],\
                                        m_prime_func[:,h_index,q_index],\
                                        t_prime_funcs,t,ret_cons = True), C_min)
                
                m_prime1         = min(max(HA_FOC(a_adj[i],H[h_index],\
                                        Q[q_index],\
                                        M[m_index],\
                                        m_prime_func[:,h_index,q_index],\
                                        t_prime_funcs,t,ret_mort = True), 0), M[m_index]*(1-amort_rate(t-2)))


                m_extra_payment  =  max(0,M[m_index]*(1-amort_rate(t-2))- m_prime1)

                wealth_endgrid[i]   = c_adj[i] + a_adj[i]+ Q[q_index]\
                        *H[h_index]\
                        *(1+tau_housing)\
                        +m_extra_payment


            elif h_index ==0: 
                a_adj[i]    = A_min

                c_adj[i]    = C_min/2

                wealth_endgrid[i]   = c_adj[i] + a_adj[i]+ Q[q_index]\
                *H[h_index]\
                *(1+tau_housing)

            else:
                a_adj[i]            = np.nan
                c_adj[i]            = np.nan
                wealth_endgrid[i]   = np.nan

        a_adj_uniform, c_adj_uniform,H_adj_uniform \
        = interp_adj(a_adj,c_adj, wealth_endgrid, extrap= True)

        return a_adj_uniform, c_adj_uniform,H_adj_uniform

    @njit
    def eval_policy_R_adj_nl(t,m_prime_func, t_prime_funcs):

        """ Generate policy functions with housing stcok adjustment
                and zero liquid saving A_t+1
        """

        wealth_endgrid_nl  = np.zeros(grid_size_H*grid_size_Q*grid_size_M)
        c_adj_nl           = np.zeros(grid_size_H*grid_size_Q*grid_size_M)
        a_adj_nl           = np.zeros(grid_size_H*grid_size_Q*grid_size_M)

        zeta               = np.zeros(grid_size_H*grid_size_Q*grid_size_M)

        for i in range(len(X_H_R_ind)):

            h_index = X_H_R_ind[i][0]
            q_index = X_H_R_ind[i][1]
            m_index = X_H_R_ind[i][2]

            args_H_FOC = (A_min,H[h_index],\
                            Q[q_index],\
                            M[m_index],\
                            t_prime_funcs,t)


            if H_FOC(C_min, *args_H_FOC)*H_FOC(C_max, *args_H_FOC)<0:

                c_a_min = max(brentq(H_FOC, C_min,C_max,\
                        args = args_H_FOC)[0], C_min)

                #print(c_a_min)

                m_prime2         = min(max(H_FOC(c_a_min, A_min,H[h_index],\
                                Q[q_index],\
                                M[m_index],\
                                t_prime_funcs, t, ret_mort = True),0),M[m_index]*(1-amort_rate(t-2)))



                UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
                UC_prime_M_RHS, UF\
                    = gen_UC_RHS(t,A_min,H[h_index],Q[q_index],m_prime2,\
                        *t_prime_funcs)


                zeta[i]     = uc(c_a_min,H[h_index],alpha_housing )/UC_prime_RHS
                a_adj_nl[i] = A_min
                c_adj_nl[i] = c_a_min
                m_extra_payment2     = max(0,M[m_index]*(1-amort_rate(t-2)) - m_prime2)

                wealth_endgrid_nl[i]   = c_adj_nl[i] + a_adj_nl[i]+ Q[q_index]\
                                                        *H[h_index]\
                                                        *(1+tau_housing)\
                                                     + m_extra_payment2
            elif h_index ==0: 

                c_adj_nl[i] = C_min/2
                a_adj_nl[i] = A_min

                wealth_endgrid_nl[i] = c_adj_nl[i] + a_adj_nl[i]+ Q[q_index]\
                *H[h_index]\
                *(1+tau_housing)


                zeta[i]        = 0
            else:
                a_adj_nl[i]            = np.nan
                c_adj_nl[i]            = np.nan
                wealth_endgrid_nl[i]   = np.nan
                zeta[i]                = np.nan

        

        zeta_nl, c_adj_uniform_nl,H_adj_uniform_nl\
         = interp_adj(zeta,c_adj_nl, wealth_endgrid_nl, extrap= False)


        return zeta_nl, c_adj_uniform_nl,H_adj_uniform_nl

    @njit 
    def gen_uf_prime(t, t_prime_funcs):

        uf_prime_1 = np.zeros(len(X_R_contgp))

        for i in  prange(len(X_R_contgp)):
            x_cont_vals             = X_R_contgp[i] 

            UC_prime_RHS, UC_prime_H_RHS, UC_prime_HFC_RHS,\
            UC_prime_M_RHS, UF\
                                    = gen_UC_RHS(t,x_cont_vals[0],\
                                                   x_cont_vals[1],x_cont_vals[2],x_cont_vals[3],\
                                                    *t_prime_funcs)

            uf_prime_1[i]           = UF
        return uf_prime_1.reshape((grid_size_A, grid_size_H, grid_size_Q, grid_size_M))

    @njit
    def gen_rhs_val_adj(t,points,
                        a_prime_adj,\
                        c_prime_adj,\
                        h_prime_adj,\
                        zeta_nl_adj,\
                        c_prime_adj_nl,\
                        h_prime_adj_nl):

        """ Retrun value of interpolated  policy
            functions for housing adjuster at
            points"""


        # liquid saving multiplier (zeta>1 no liq. saving.)

        zeta_nl_val             = eval_linear(X_QH_R,\
                                    zeta_nl_adj,\
                                    points,\
                                    xto.LINEAR)

        zeta_ind                = zeta_nl_val>1

        # policies with liquid saving 

        H_prime_adj_val         = eval_linear(X_QH_R,\
                                    h_prime_adj,\
                                    points,\
                                    xto.LINEAR)

        H_prime_adj_val[H_prime_adj_val<H_min] = H_min


        c_prime_adj_val         = eval_linear(X_QH_R,\
                                        c_prime_adj,\
                                        points, xto.LINEAR)

        c_prime_adj_val[c_prime_adj_val<C_min] = C_min

        a_prime_adj_val         = eval_linear(X_QH_R,\
                                        a_prime_adj,\
                                        points,\
                                        xto.LINEAR)

        a_prime_adj_val[a_prime_adj_val<A_min] = A_min

        extra_pay_adj_val       = points[:,2]  - c_prime_adj_val\
                                               - H_prime_adj_val\
                                                    *(1+tau_housing)\
                                                    *points[:,1]\
                                               - a_prime_adj_val

        extra_pay_adj_ind       = extra_pay_adj_val> 1e-05

        mort_dp_prime           = points[:,1]*(1-amort_rate(t-2))\
                                                - extra_pay_adj_val

        mort_dp_prime[mort_dp_prime<=0] = 0


        # policies without liquid saving 

        H_prime_adj_nl_val      = eval_linear(X_QH_R,h_prime_adj_nl,\
                                                points,\
                                                xto.LINEAR)

        H_prime_adj_nl_val[H_prime_adj_val<0] = H_min

        c_prime_adj_nl_val      = eval_linear(X_QH_R,\
                                                c_prime_adj_nl,\
                                                points, xto.LINEAR)

        c_prime_adj_nl_val[c_prime_adj_val<C_min] = C_min

        extra_pay_val_nl        = points[:,2] - c_prime_adj_nl_val\
                                              - H_prime_adj_nl_val\
                                                *(1+tau_housing)\
                                                *points[:,1]

        extra_pay_adj_nl_ind    = extra_pay_val_nl >1e-05

        mort_dp_prime_nl        = points[:,1]*(1-amort_rate(t-2))\
                                             - extra_pay_val_nl

        mort_dp_prime_nl[mort_dp_prime_nl<0] = 0

        # combine pols for non-renter          

        H_prime_val             = (1-zeta_ind)*H_prime_adj_val \
                                    + zeta_ind*H_prime_adj_nl_val

        c_prime_val             = (1-zeta_ind)*c_prime_adj_val\
                                    + zeta_ind*c_prime_adj_nl_val

        a_prime_val             = (1-zeta_ind)*a_prime_adj_val

        extra_pay_ind           = (1-zeta_ind)*extra_pay_adj_ind\
                                    + zeta_ind*extra_pay_adj_nl_ind 

        mort_dp_prime           = (1-zeta_ind)*mort_dp_prime\
                                    + zeta_ind*mort_dp_prime_nl 

        return c_prime_val,H_prime_val, a_prime_val,\
                    mort_dp_prime, extra_pay_ind


    @njit
    def gen_rhs_val_noadj(t,points,
                            a_prime_noadj,\
                            c_prime_noadj,\
                            eta_prime_noadj):

        """ Interpolate value of interped policy
            functions for housing non-adjuster at
            points"""

        etavals                   = eval_linear(X_cont_R,\
                                                    eta_prime_noadj,\
                                                    points,\
                                                    xto.LINEAR)

        H_prime_noadj_val         = points[:,1]*(1-delta_housing)
        
        c_prime_noadj_val         = eval_linear(X_cont_R,\
                                                c_prime_noadj,\
                                                points,\
                                                xto.LINEAR)

        c_prime_noadj_val[c_prime_noadj_val<C_min] = C_min

        a_prime_noadj_val         = eval_linear(X_cont_R,\
                                                a_prime_noadj,\
                                                points,\
                                                xto.LINEAR)

        a_prime_noadj_val[a_prime_noadj_val<0] = 0

        extra_pay_noadj_val      = points[:,0] -  a_prime_noadj_val\
                                               -  c_prime_noadj_val

        extra_pay_noadj_ind      = extra_pay_noadj_val> 0

        mort_db_prime_noadj      = points[:,3]*(1-amort_rate(t-2))\
                                              -  extra_pay_noadj_val

        mort_db_prime_noadj[mort_db_prime_noadj<0] = 0

        return c_prime_noadj_val,H_prime_noadj_val, etavals,\
                mort_db_prime_noadj,extra_pay_noadj_ind

    @njit
    def gen_rhs_val_rent(t,points,
                    h_prime_rent):

        """ Interpolate value of interped policy
            functions for housing non-adjuster at
            points"""

        h_prime_rent_val        = eval_linear(W_Q_R,\
                                                h_prime_rent,\
                                                points,\
                                                xto.LINEAR)

        c_prime_rent_val        = phi_r*points[0,1]*h_prime_rent_val\
                                    *(1-alpha_housing)/alpha_housing

        c_prime_rent_val[c_prime_rent_val<=C_min] = C_min
        h_prime_rent_val[h_prime_rent_val<=H_min] = H_min

        return c_prime_rent_val,h_prime_rent_val


    #@njit(parallel = True)    
    def gen_RHS_TR(t, a_prime_noadj,\
                    c_prime_noadj,\
                    eta_prime_noadj,\
                    a_prime_adj,\
                    c_prime_adj,\
                    h_prime_adj,\
                    zeta_nl,\
                    c_prime_adj_nl,\
                    h_prime_adj_nl,\
                    h_prime_rent,\
                    UF_dbprime):

        """Generate RHS T_R Euler equation conditioned on:
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
        
        """

        UC_prime_out        = np.zeros(len(X_all_hat_ind))
        UC_prime_H_out      = np.zeros(len(X_all_hat_ind))
        UC_prime_HFC_out    = np.zeros(len(X_all_hat_ind))
        UC_prime_M_out      = np.zeros(len(X_all_hat_ind))
        Lambda_out          = np.zeros(len(X_all_hat_ind))
        VF                  = np.zeros(len(X_all_hat_ind))


        # array of possible DB pay-outs for this age
        DB_payout = np.zeros(len(E))

        for i in range(len(E)):
            DB_payout[i] = DB_benefit(t, t-tzero,\
                            y(t, E[i]),\
                            i,\
                            P_E,\
                            P_stat,\
                            E) 
        @njit  
        def gen_RHS_TR_point(points,\
                                UC_prime_out,\
                                UC_prime_H_out,\
                                UC_prime_HFC_out,\
                                UC_prime_M_out,\
                                Lambda_out,VF) :
            """
            Loop over states, where each i
            indexes a cartesian product of:

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
            for i in prange(len(points)):
                q_in                = X_all_hat_vals[i][8]
                H_in                = X_all_hat_vals[i][7]
                q_ind               = X_all_hat_ind[i][8]
                E_ind               = X_all_hat_ind[i][1]
                ADC_in              = X_all_hat_vals[i][6]
                r_share             = X_all_hat_vals[i][4]
                m_in                = X_all_hat_vals[i][9]

                # generate values for relisations of T_R period
                # house price shocks, DC values after returns
                # mortgage interest shocks and mortgage balances
                # after interest 

                Q_prime             = q_in*(1+r_H + Q_DC_shocks[:,2])

                A_DC_prime          = (1+(1-r_share)*Q_DC_shocks[:,0]\
                                        + r_share*Q_DC_shocks[:,1] )*ADC_in

                r_m_prime           = beta_m*r_l\
                                        *(Q_DC_shocks[:,0]/r_l)**kappa_m

                M_prime             = (1+r_m_prime)*m_in  

                # for each T_R-1 period exogenous state, 
                # loop over R period wage stock realisation
                for j in prange(len(E)):
                    
                    a_l_exDC                = DB_payout[j]\
                                                *(1-X_all_hat_vals[i][0])\
                                                + (1+r)*X_all_hat_vals[i][5]

                    a_l                     = A_DC_prime+ a_l_exDC \
                                                - M_prime*amort_rate(R-2)

                    wealth                  = a_l + Q_prime*H_in\
                                                    *(1-delta_housing)
                    wealth[wealth<=0]       = 0

                    A_prime                 = wealth - M_prime
                    A_prime[A_prime<=0]     =  1E-100

                    h_prime_arr             = np.full(len(Q_prime),\
                                                (1-delta_housing)*H_in)
                    
                    # state points of length housing price shock x DC shock 
                    # interpolation policy functions over these points 

                    point_noadj     = np.column_stack((a_l,\
                                        h_prime_arr, Q_prime,M_prime))

                    points_adj      = np.column_stack((Q_prime,\
                                                     M_prime, wealth))

                    points_rent     = np.column_stack((Q_prime,wealth \
                                        - M_prime*(1-amort_rate(R+1-2)\
                                        - tau_housing*Q_prime*H_in)))

                    points_rent[:,2][points_rent[:,2]<=0]  = 0

                    c_prime_val_noadj,H_prime_val_noadj, etavals,\
                    mort_db_prime_noadj,extra_pay_ind_noadj =\
                                        gen_rhs_val_noadj(t,point_noadj,
                                                    a_prime_noadj,\
                                                    c_prime_noadj,\
                                                    eta_prime_noadj)

                    c_prime_val_adj,H_prime_val_adj, a_prime_val_adj,\
                    mort_dp_prime_adj,extra_pay_ind_adj     =\
                                        gen_rhs_val_adj(t,points_adj,
                                                    a_prime_adj,\
                                                    c_prime_adj,\
                                                    h_prime_adj,\
                                                    zeta_nl,\
                                                    c_prime_adj_nl,\
                                                    h_prime_adj_nl)
                    
                    c_prime_val_rent,h_prime_val_rent  = \
                                        gen_rhs_val_rent(t,points_rent,
                                                          h_prime_rent)

                    eta_ind         = (extra_pay_ind_noadj<= 1)*(a_l>=0)  


                    # combine non-renting policies

                    c_prime_val     = eta_ind*c_prime_val_noadj\
                                        + (1-eta_ind)*c_prime_val_adj
                    H_prime_val     = eta_ind*H_prime_val_noadj\
                                        + (1-eta_ind)*H_prime_val_adj
                    a_prime_val     = eta_ind*H_prime_val_noadj\
                                        + (1-eta_ind)*a_prime_val_adj
                    mort_db_prime   = eta_ind*mort_db_prime_noadj\
                                        + (1-eta_ind)*mort_dp_prime_adj
                    exrtra_pay_norent_ind   = eta_ind*extra_pay_ind_noadj\
                                        + (1-eta_ind)*extra_pay_ind_adj

                    uc_prime_norent = uc(c_prime_val,\
                                            H_prime_val,\
                                            alpha_housing)

                    u_norent        = u(c_prime_val,\
                                            H_prime_val,\
                                            alpha_housing)

                    state_dp_prime_norent   = np.column_stack((a_prime_val,\
                                                         H_prime_val,\
                                                         Q_prime,\
                                                         mort_db_prime))

                    UF_dp_val_norent  = beta_bar*eval_linear(X_cont_R,\
                                                UF_dbprime,\
                                                state_dp_prime_norent )

                    # policies with renting 

                    uc_prime_rent        = uc(c_prime_val_rent,\
                                                h_prime_val_rent,\
                                                alpha_housing)

                    u_rent               = u(c_prime_val_rent,\
                                                c_prime_val_rent,\
                                                alpha_housing)

                    a_prime_rent_val     = points_rent[:,1] - c_prime_val_rent\
                                                - h_prime_val_rent\
                                                    *phi_r*Q_prime

                    a_prime_rent_val[a_prime_rent_val<=A_min] = A_min

                    state_dp_prime_rent  = np.column_stack((a_prime_rent_val,\
                                                np.full(len(a_prime_rent_val), H_min),\
                                                Q_prime, np.full(len(a_prime_rent_val), 0)))

                    UF_dp_val_rent       = beta_bar*eval_linear(X_cont_R,\
                                                    UF_dbprime,\
                                                    state_dp_prime_rent)
                    # index to rent or not

                    rent_ind             = u_rent+ UF_dp_val_rent> \
                                            u_norent+UF_dp_val_norent

                    # combine R period marginal utilities and utility value
                    # across renters and non-renters 

                    uc_prime            = rent_ind*uc_prime_rent \
                                            + (1-rent_ind)*uc_prime_norent
                    exrtra_pay_ind      = rent_ind*1 \
                                            + (1-rent_ind)*exrtra_pay_norent_ind

                    uf                  = rent_ind*u_rent \
                                            + (1-rent_ind)*u_norent

                    # generate combined marginal utilities 
                    # wrt liq. assets, housing, adjusting housing, mortages
                    # DC assets and R period utility value 
                    # 
                    # we have multiplied functions with probability of wage shock for state j
                    # conditioned on state E_ind in the previous period 
                    # *note we sum over the j wage shock probs in the loop over len(Q_DC_P)*


                    UC_prime        =   P_E[E_ind][j]*(1+r)*(s[int(R-1)]*uc_prime\
                                        + (1-s[int(R-1)])*b_prime(A_prime))                         # question: should wealth here in the bequest function include or not include 
                                                                                                    # the DC and DB pay-out?
                                                                                                    # shouldnt the DB payout go in every year?                
                    UC_prime_H      =    P_E[E_ind][j]*Q_prime*(1-delta_housing - tau_housing*rent_ind)*(\
                                            s[int(R-1)]*uc_prime
                                            + (1-s[int(R-1)])*b_prime(A_prime))

                    UC_prime_HFC    =    P_E[E_ind][j]*s[int(R-1)]*eta_ind*(1-rent_ind)*uc_prime*\
                                            Q_prime*tau_housing*H_prime_val                             # question: should the adjustment 
                                                                                                        # cost function come under the bequest?
                    
                    UC_prime_M     =    P_E[E_ind][j]*(1+r_m_prime)\
                                            *(s[int(R-1)]*uc_prime*((exrtra_pay_ind))\
                                            + (1-s[int(R-1)])*b_prime(A_prime))

                    Lambda         =    (1+(1-r_share)*Q_DC_shocks[:,0]\
                                            + r_share*Q_DC_shocks[:,1])*\
                                            (UC_prime/(1+r))

                    VF_cont        =    s[int(R-1)]*P_E[E_ind][j]*uf +\
                                        (1-s[int(R-1)])*b(A_prime)

                    for n in prange(len(Q_DC_P)):
                    # gen UC_prime unconditioned on T_R -1 income
                        UC_prime_out[i]         += Q_DC_P[n]*UC_prime[n]
                        
                        UC_prime_H_out[i]       += Q_DC_P[n]*UC_prime_H[n]

                        UC_prime_HFC_out[i]     += Q_DC_P[n]*UC_prime_HFC[n]

                        UC_prime_M_out[i]       += Q_DC_P[n]*UC_prime_M[n]

                        Lambda_out[i]           += Q_DC_P[n]*Lambda[n] 

                        VF[i]                   += Q_DC_P[n]*VF_cont[n]

            return UC_prime_out,UC_prime_H_out, UC_prime_HFC_out,UC_prime_M_out,\
                    Lambda_out, VF 


        UC_prime_out,UC_prime_H_out, UC_prime_HFC_out,UC_prime_M_out,\
        Lambda_out, VF              = gen_RHS_TR_point(np.arange(len(X_all_hat_vals)),\
                                                UC_prime_out,\
                                                UC_prime_H_out,\
                                                UC_prime_HFC_out,\
                                                UC_prime_M_out,Lambda_out,VF)
    
        return UC_prime_out, UC_prime_H_out, UC_prime_HFC_out, UC_prime_M_out, Lambda_out,VF

    #@njit
    def gen_R_pol():

        t_prime_funcs   = (np.ones((grid_size_A,grid_size_H,grid_size_Q,grid_size_M)),\
                            np.ones((grid_size_A,grid_size_H,grid_size_Q,grid_size_M)),\
                            np.ones((grid_size_A,grid_size_H,grid_size_Q,grid_size_M)),\
                            np.ones((grid_size_Q,grid_size_M, grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M,grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M, grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M, grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M,grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M, grid_size_A)),\
                            np.ones((grid_size_A, grid_size_Q)),\
                            np.zeros((grid_size_A,grid_size_H,grid_size_Q,grid_size_M)))

        UF_dbprime        = gen_uf_prime(T, t_prime_funcs)

        t_prime_funcs   = (np.ones((grid_size_A,grid_size_H,grid_size_Q,grid_size_M)),\
                            np.ones((grid_size_A,grid_size_H,grid_size_Q,grid_size_M)),\
                            np.ones((grid_size_A,grid_size_H,grid_size_Q,grid_size_M)),\
                            np.ones((grid_size_Q,grid_size_M, grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M,grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M, grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M, grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M,grid_size_A)),\
                            np.ones((grid_size_Q,grid_size_M, grid_size_A)),\
                            np.ones((grid_size_A, grid_size_Q)),\
                            UF_dbprime)

        for i in range(int(T-R+1)):
            t = T-i
            print(t)
            #start = time.time() 
            h_prime_rent                                 = eval_rent_pol(t_prime_funcs,t)
            #print(time.time() - start)
            #start = time.time()
            m_prime_func                                 = eval_mort_policy(t,t_prime_funcs)    
            #print(time.time() - start)
            #start = time.time()       
            a_noadj, c_noadj, etas_noadj                 = eval_policy_R_noadj(t, m_prime_func,t_prime_funcs) 
            #rint(time.time() - start)
            #start = time.time()
            a_adj_uniform, c_adj_uniform,H_adj_uniform,\
               = eval_policy_R_adj(t,m_prime_func, t_prime_funcs) 

            zeta_nl, c_adj_uniform_nl,H_adj_uniform_nl = eval_policy_R_adj_nl(t,m_prime_func, t_prime_funcs) 
            #print(time.time() - start)

            UF_dbprime= gen_uf_prime(t, t_prime_funcs)
            #print(UF_dbprime)
            t_prime_funcs =\
            (a_noadj, c_noadj, etas_noadj, a_adj_uniform, c_adj_uniform,\
             H_adj_uniform, zeta_nl, c_adj_uniform_nl,H_adj_uniform_nl, h_prime_rent, UF_dbprime)

        #start = time.time()
        UC_prime_out, UC_prime_H_out, UC_prime_HFC_out,UC_prime_M_out,Lambda,VF = gen_RHS_TR(t, *t_prime_funcs)
        #UC_prime_out, UC_prime_H_out, UC_prime_HFC_out,UC_prime_M_out,Lambda,VF = 0,0,0,0,0,0
        #print(time.time() - start)
        return a_noadj,c_noadj, etas_noadj,a_adj_uniform,\
                c_adj_uniform,H_adj_uniform,zeta_nl, c_adj_uniform_nl,\
                H_adj_uniform_nl, h_prime_rent,UC_prime_out, UC_prime_H_out,\
                UC_prime_HFC_out,UC_prime_M_out, Lambda,VF
    
    return gen_R_pol



if __name__ == "__main__":


    import numpy as np
    import pandas as pd
    from numpy import genfromtxt
    import csv
    import numba 
    from quantecon import tauchen
    import matplotlib.pyplot as plt
    from matplotlib.colors import DivergingNorm
    import time
    from pathos.multiprocessing import ProcessingPool
    import dill as pickle 
    from randparam   import rand_p_generator
    import copy
    import sys
    from housing_functions import housingmodel_function_factory
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import pandas as pd
    from egg_basket import HousingModel


    normalisation = np.array([1, 1E-5])
    param_deterministic = {}
    param_random_bounds = {}
    N = 480

    settings_folder = '/home/141/as3442/Retirementeggs/settings'

    with open('{}/parameters_EGM_base.csv'.format(settings_folder),\
        newline='') as pscfile:
        reader = csv.DictReader(pscfile)
        for row in reader:
            param_deterministic[row['parameter']] = np.float64(row['value'])

    with open('{}/random_param_bounds.csv'\
        .format(settings_folder), newline='') as pscfile:
        reader_ran = csv.DictReader(pscfile)
        for row in reader_ran:
            print(row['UB'])
            param_random_bounds[row['parameter']] = np.float64([row['LB'],\
                row['UB']])

    lambdas             = genfromtxt('{}/lambdas_male.csv'.\
                            format(settings_folder), delimiter=',')[1:]
    survival            = genfromtxt('{}/survival_probabilities_male.csv'.\
                            format(settings_folder), delimiter=',')[0:]
    vol_cont_points     = genfromtxt('{}/vol_cont_points.csv'.\
                            format(settings_folder), delimiter=',')[1:]
    risk_share_points   = genfromtxt('{}/risk_share_points.csv'.\
                            format(settings_folder), delimiter=',')[1:]
    
    parameters          = rand_p_generator(param_deterministic,\
                             param_random_bounds, deterministic = 1,initial =1)

    #load latest estimates
    estimates = pickle.load(open("/scratch/pv33/latest_means_iter.smms","rb")) 

     #parameters['gamma'] = .3


    for i,key in zip(np.arange(len(estimates[0])),param_random_bounds.keys()):
        parameters[key]  = estimates[0][i]


    #parameters['alpha_bar'] = .6
    #parameters['gamma'] = 3.5
    parameters['beta_bar'] = .95

    # prepare the data moments 
    moments_data = pd.read_csv('{}/moments_data.csv'.format(settings_folder))
    moments_data = moments_data.drop('Unnamed: 0', axis=1)   
    moments_data_array   = np.array(np.ravel(moments_data.iloc[:,1:115]))

    functions = {}

    functions['u'], functions['uc'], functions['uh'], functions['b'], \
    functions['b_prime'], functions['y'],functions['yvec'], functions['DB_benefit'], \
    functions['adj_p'], functions['adj_v'], functions['adj_pi'],\
    functions['uc_inv'],functions['uh_inv'], functions['amort_rate']\
        = housingmodel_function_factory(parameters,\
                                         lambdas,\
                                          normalisation)
    
    # Create housing model 
    og              = HousingModel(functions, parameters, survival,\
                                        vol_cont_points,\
                                        risk_share_points, np.array([1]))

    gen_R_pol      = housing_model_retiree_func_factory(og)


    a_noadj, c_noadj, etas_noadj, a_adj_uniform, c_adj_uniform,\
    H_adj_uniform,zeta_nl, c_adj_uniform_nl,H_adj_uniform_nl, h_prime_rent,UC_prime_out, UC_prime_H_out,\
    UC_prime_HFC_out,UC_prime_M_out, Lambda,VF= gen_R_pol()      


    consumption                     = np.empty((len(og.M), len(og.A_R)))
    assets_prime                    = np.empty((len(og.M), len(og.A_R)))
    H_prime                         = np.empty((len(og.M), len(og.A_R)))

    

    #ax = plt.axes(0,2)
    for k in range(len(og.Q)):
        for j in range(len(og.H)):
            plt.close()
            NUM_COLORS  = len(og.M)
            colormap    = cm.viridis
            fig, ax     = plt.subplots(3,2,figsize =(11.69,8.27))
            M_vals      = np.linspace(0,4, 15)
            normalize   = mcolors.Normalize(vmin=np.min(M_vals), vmax=np.max(M_vals))
            for i in range(len(og.M)-1):
                mort    = og.M[i]

                h_index = j
                q_index = k
                h       = round(og.H[h_index]*(1-og.delta_housing), 0)
                q       = og.Q[q_index]
                wealth              = og.A_R + q*h 

                consumption_adj_l     = interp(og.W_R, c_adj_uniform[q_index,i, :], og.W_R)
                assets_adj_l           = interp(og.W_R, a_adj_uniform[q_index,i, :], og.W_R)
                H_adj_l                = interp(og.W_R, H_adj_uniform[q_index,i, :], og.W_R)

                consumption_adj_nl     = interp(og.W_R, c_adj_uniform_nl[q_index,i, :], og.W_R)
                zeta_nl_adj            = interp(og.W_R, zeta_nl[q_index,i, :], og.W_R)
                zeta_adj_nl_ind         = zeta_nl_adj>1
                H_adj_nl               = interp(og.W_R, H_adj_uniform_nl[q_index,i, :], og.W_R)

                consumption_adj         = zeta_adj_nl_ind*consumption_adj_nl + (1-zeta_adj_nl_ind)*consumption_adj_l
                H_adj                   = zeta_adj_nl_ind*H_adj_nl + (1-zeta_adj_nl_ind)*H_adj_l
                assets_adj              = zeta_adj_nl_ind*og.A_min + (1-zeta_adj_nl_ind)*assets_adj_l

                payment_adj         = wealth -consumption_adj-  assets_adj - H_adj*q*(1+og.tau_housing)

                etas                = etas_noadj[:, h_index,q_index, i ]
                consumption_noadj   = c_noadj[:, h_index,q_index, i ]
                assets_noadj        = a_noadj[:, h_index,q_index, i ]

                payment_noadj       = og.A_R - consumption_noadj- assets_noadj

                min_payment         = og.amort_rate(64-2)*mort

                etas2               = np.abs(etas)>1 

                #etas2               = etas1>=1

                consumption[i]      = etas2*consumption_adj + (1-etas2)*consumption_noadj
                assets_prime[i]     = etas2*assets_adj      + (1-etas2)*assets_noadj
                H_prime[i]          = etas2*H_adj           + (1-etas2)*h

                payment_actual      = etas2*payment_adj     + (1-etas2)*payment_noadj

                points              = np.column_stack((assets_prime[i],H_prime[i], np.full(len(assets_prime[i]), q),consumption[i]))


                #print(mortgage_pay)

                #mortgage_pay[mortgage_pay<=min_payment] = min_payment

                mortgage_pay2 = og.W_R- c_adj_uniform[k,i, :] - a_adj_uniform[k,i, :] - H_adj_uniform[k,i, :]*q*(1+og.tau_housing)
                # mortgage_pay[mortgage_pay >= wealth]        =wealth[mortgage_pay >= wealth] 

                ax[0,0].plot(og.W_R ,H_adj,color = colormap(i//3*3.0/NUM_COLORS))
                ax[0,0].set_xlabel('Total liquid wealth (after adjusting)')
                ax[0,0].set_ylabel('Housing  (if adjusting)')
                #ax[0,0].set_title('Housing policy (after adjusting)')

                ax[0,1].plot(og.W_R,assets_adj,color = colormap(i//3*3.0/NUM_COLORS))
                ax[0,1].set_xlabel('Total liquid wealth (after adjusting)')
                ax[0,1].set_ylabel('Assts (if adjusting)')
                #ax[0,1].set_title('Cons policy (after adjusting)')

                ax[1,0].plot(wealth ,H_prime[i],color = colormap(i//3*3.0/NUM_COLORS))
                ax[1,0].set_xlabel('Total wealth (liquid and illquid)')
                ax[1,0].set_ylabel('Actual housing')
                #ax[1,0].set_title('Housing policy for real wealth {}'.format(int(h)))

                ax[1,1].plot(wealth,consumption[i],color = colormap(i//3*3.0/NUM_COLORS))
                ax[1,1].set_xlabel('Total wealth (liquid and illquid)')
                ax[1,1].set_ylabel('Actual consumption')

                ax[2,0].plot(og.W_R,mortgage_pay2,color = colormap(i//3*3.0/NUM_COLORS))
                ax[2,0].set_xlabel('Consumption')
                ax[2,0].set_ylabel('Mortgage payment')

                ax[2,1].plot(og.A_R,payment_noadj,color = colormap(i//3*3.0/NUM_COLORS))
                ax[2,1].set_xlabel('Total wealth (liquid and illquid)')
                ax[2,0].set_ylabel('Mortgage payment')
                #ax[2,1].set_title('Mort payment for real wealth of {}'.format(int(h)))

            #plt.plot(wealth,etas2)
            # setup the colorbar
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(M_vals)
            cbar = plt.colorbar(scalarmappaple)
            cbar.set_label('Mort. liability')
            plt.tight_layout()
            plt.savefig('mortgage_retiree_H{}_Q{}.png'.format(j,k))

