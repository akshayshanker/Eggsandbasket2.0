import yaml
import dill as pickle
import glob 
import copy
import os
import time
import gc

# Housing model modules
import lifecycle_model 
from solve_policies.worker_solver import generate_worker_pols
from generate_timeseries.tseries_generator import gen_panel_ts, gen_moments, sortmoments, genprofiles_operator

import numpy as np

# Read settings
with open("settings/settings.yml", "r") as stream:
    eggbasket_config = yaml.safe_load(stream)


og = lifecycle_model.LifeCycleModel(eggbasket_config['baseline_lite'],
                np.array([0]), param_id = 'test', mod_name = 'test')

from mpi4py import MPI as MPI4py
world = MPI4py.COMM_WORLD


og.ID = 'CNX7HO_20201103-165519_test'
modname = 'test'

generate_TSDF,load_pol_array = genprofiles_operator(og)
policy = load_pol_array(og.ID,modname)



from matplotlib.colors import DivergingNorm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plots_folder = '/home/141/as3442/temp_plots/'
NUM_COLORS = len(og.grid1d.M)
colormap = cm.viridis
M_vals = np.linspace(0, og.grid1d.M[-1], 9)
normalize = mcolors.Normalize(vmin=np.min(M_vals), vmax=np.max(M_vals))

acc_ind = 1

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

            plt.savefig('{}/{}_W_{}_DC{}_P{}.png'.format(plots_folder, age, key, k, l))

            plt.close()

"""



TSN = 100
U = np.random.rand(6,100,TSN,100) 

TSALL_10_df, TSALL_14_df = gen_panel_ts(og,U, TSN)

moments_male = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_male') 

moments_female = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_female')


moments_sim_sorted = sortmoments(moments_male,\
                                     moments_female)


# Return policies 

"""
