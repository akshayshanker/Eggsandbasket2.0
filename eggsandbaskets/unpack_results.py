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
with open("settings.yml", "r") as stream:
    eggbasket_config = yaml.safe_load(stream)


og = lifecycle_model.LifeCycleModel(eggbasket_config['baseline_lite'],
                np.array([0]), param_id = 'test', mod_name = 'test')

from mpi4py import MPI as MPI4py
world = MPI4py.COMM_WORLD

#ID = '20201023-132142_test_0'

#policies = generate_worker_pols(og,world, load_retiree=1, gen_newpoints = False)

#pickle.dump(policies, \
#           open("/scratch/pv33/ls_model_temp/DB_lite.pols", "wb"))


ID = 'testJUT4BB_20201025-150000_test'

numpy_vars_DC = {}
numpy_vars_DB = {}
os.chdir('/scratch/pv33/ls_model_temp/{}'.format(ID+'_acc_'+str(1)))
for np_name in glob.glob('*np[yz]'):
    numpy_vars_DC[np_name] = dict(np.load(np_name, mmap_mode = 'r'))

os.chdir('/scratch/pv33/ls_model_temp/{}'.format(ID+'_acc_'+str(0)))
for np_name in glob.glob('*np[yz]'):
    numpy_vars_DB[np_name] = dict(np.load(np_name, mmap_mode = 'r'))

var_keys = copy.copy(list(numpy_vars_DB.keys()))

for keys in var_keys:
    numpy_vars_DB[keys.split('_')[1]] = numpy_vars_DB.pop(keys)

var_keys = copy.copy(list(numpy_vars_DC.keys()))
for keys in var_keys:
    numpy_vars_DC[keys.split('_')[1]] = numpy_vars_DC.pop(keys)

#npz_file_dict = np.load("/scratch/pv33/ls_model_temp")
policy_c_noadj = []
etas_noadj = []
policy_a_noadj = []
policy_c_adj = []
policy_h_adj = []
policy_a_adj = []
policy_h_rent = []
policy_zeta = []
policy_prob_v = []
policy_prob_pi = []
tzero = og.parameters.tzero
R = og.parameters.R

for Age in np.arange(int(og.parameters.tzero), int(og.parameters.R)):

        start = time.time()
        policy_c_adj.append(np.concatenate((numpy_vars_DB[str(int(Age))]['C_adj'], numpy_vars_DC[str(int(Age))]['C_adj'])))
        policy_h_adj.append(np.concatenate((numpy_vars_DB[str(int(Age))]['H_adj'], numpy_vars_DC[str(int(Age))]['H_adj'])))
        policy_a_adj.append(np.concatenate((numpy_vars_DB[str(int(Age))]['Aprime_adj'], numpy_vars_DC[str(int(Age))]['Aprime_adj'])))
        policy_c_noadj.append(np.concatenate((numpy_vars_DB[str(int(Age))]['C_noadj'],numpy_vars_DC[str(int(Age))]['C_noadj'])))
        etas_noadj.append(np.concatenate((numpy_vars_DB[str(int(Age))]['etas_noadj'], numpy_vars_DC[str(int(Age))]['etas_noadj'])))
        policy_a_noadj.append(np.concatenate((numpy_vars_DB[str(int(Age))]['Aprime_noadj'], numpy_vars_DC[str(int(Age))]['Aprime_noadj'])))
        policy_zeta.append(np.concatenate((numpy_vars_DB[str(int(Age))]['zeta'], numpy_vars_DC[str(int(Age))]['zeta'])))
        policy_h_rent.append(np.concatenate((numpy_vars_DB[str(int(Age))]['H_rent'], numpy_vars_DC[str(int(Age))]['H_rent'])))
        policy_prob_v.append(np.concatenate((numpy_vars_DB[str(int(Age))]['prob_v'], numpy_vars_DC[str(int(Age))]['prob_v'])))
        policy_prob_pi.append(np.concatenate((numpy_vars_DB[str(int(Age))]['prob_pi'], numpy_vars_DC[str(int(Age))]['prob_pi'])))

        print("Loaded policies for DB age {} in {}".format(Age, time.time()-start))

        if Age== og.parameters.tzero:
            policy_VF = np.concatenate((numpy_vars_DB[str(int(Age))]['policy_VF'],numpy_vars_DC[str(int(Age))]['policy_VF']))
        del numpy_vars_DB[str(int(Age))]
        del numpy_vars_DC[str(int(Age))]
        gc.collect()

policy = [policy_c_noadj,\
                    etas_noadj,\
                    policy_a_noadj,\
                    policy_c_adj,\
                    policy_h_adj,\
                    policy_a_adj,\
                    policy_h_rent,\
                    policy_zeta,\
                    policy_prob_v,\
                    policy_prob_pi,\
                    policy_VF]

from matplotlib.colors import DivergingNorm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plots_folder = '/home/141/as3442/temp_plots/'
NUM_COLORS = len(og.grid1d.M)
colormap = cm.viridis
M_vals = np.linspace(0, og.grid1d.M[-1], 15)
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

            plt.savefig(
                '{}/{}_W_{}_DC{}_P{}.png'.format(plots_folder, age, key, k, l))

            plt.close()

"""

TSN = 100
U = np.random.rand(6,100,TSN,100) 

TSALL_10_df, TSALL_14_df = gen_panel_ts(og, ID,U, TSN)

moments_male        = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_male') 

moments_female      = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_female')


moments_sim_sorted    = sortmoments(moments_male,\
                                     moments_female)
"""



