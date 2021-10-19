import yaml
import dill as pickle
import glob 
import copy
import os
import time
import warnings
from interpolation import interp

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm


# Housing model modules
import lifecycle_model 
from solve_policies.worker_solver import generate_worker_pols
from generate_timeseries.tseries_generator import gen_panel_ts,\
					 gen_moments, genprofiles_operator
from util.helper_funcs import get_jobfs_path, read_settings
from generate_timeseries.ts_helper_funcs import sortmoments
from generate_timeseries.plot_ts import plot_moments_ts
from smm import gen_communicators

def plot_retiree(pols, og):

	consumption = np.empty((len(og.grid1d.M), len(og.grid1d.A_R)))
	assets_prime = np.empty((len(og.grid1d.M), len(og.grid1d.A_R)))
	H_prime = np.empty((len(og.grid1d.M), len(og.grid1d.A_R)))

	a_noadj, c_noadj, a_adj_uniform, c_adj_uniform,\
	H_adj_uniform, h_prime_rent, UC_prime_out, UC_prime_H_out, UC_prime_M_out, VF = pols

	#ax = plt.axes(0,2)
	for k in range(len(og.grid1d.Q)):
		for j in range(len(og.grid1d.H)):
			plt.close()
			NUM_COLORS = len(og.grid1d.M)
			colormap = cm.viridis
			fig, ax = plt.subplots(3, 2, figsize=(11.69, 8.27))
			M_vals = np.linspace(0, .8, 15)
			normalize = mcolors.Normalize(
				vmin=np.min(M_vals), vmax=np.max(M_vals))
			for i in range(len(og.grid1d.M) - 1):
				mort = og.grid1d.M[i]

				h_index = j
				q_index = k
				h = round(og.grid1d.H[h_index] * (1 - og.parameters.delta_housing), 0)
				q = og.grid1d.Q[q_index]
				wealth = og.grid1d.A_R + q * h

				consumption_adj = interp(og.grid1d.W_R, c_adj_uniform[q_index, i, :], og.grid1d.W_R)
				assets_adj = interp(
					og.grid1d.W_R, a_adj_uniform[q_index, i, :], og.grid1d.W_R)
				H_adj = interp(og.grid1d.W_R, H_adj_uniform[q_index, i, :], og.grid1d.W_R)

				consumption_noadj = c_noadj[:, h_index, q_index, i]
				assets_noadj = a_noadj[:, h_index, q_index, i]


				ax[0, 0].plot(og.grid1d.W_R, H_adj, color=colormap(
					i // 3 * 3.0 / NUM_COLORS))
				ax[0, 0].set_xlabel('Total liquid wealth (after adjusting)')
				ax[0, 0].set_ylabel('Housing  (if adjusting)')
				#ax[0,0].set_title('Housing policy (after adjusting)')

				ax[0, 1].plot(og.grid1d.W_R, consumption_adj, color=colormap(
					i // 3 * 3.0 / NUM_COLORS))
				ax[0, 1].set_xlabel('Total liquid wealth (after adjusting)')
				ax[0, 1].set_ylabel('Consumption (if adjusting)')
				#ax[0,1].set_title('Cons policy (after adjusting)')

				ax[1, 0].plot(og.grid1d.A_R, assets_noadj, color=colormap(
					i // 3 * 3.0 / NUM_COLORS))
				ax[1, 0].set_xlabel('Total wealth (liquid and illquid)')
				ax[1, 0].set_ylabel('Assets no adjustibg')
				#ax[1,0].set_title('Housing policy for real wealth {}'.format(int(h)))

				ax[1, 1].plot(og.grid1d.A_R, consumption_noadj,
							  color=colormap(i // 3 * 3.0 / NUM_COLORS))
				ax[1, 1].set_xlabel('Total wealth (liquid and illquid)')
				ax[1, 1].set_ylabel('Actual consumption no adjusting')

				#ax[2,1].set_title('Mort payment for real wealth of {}'.format(int(h)))

			# plt.plot(wealth,etas2)
			# setup the colorbar
			scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
			scalarmappaple.set_array(M_vals)
			cbar = plt.colorbar(scalarmappaple)
			cbar.set_label('Mort. liability')
			plt.tight_layout()
			plt.savefig('plots/mortgage_retiree_H{}_Q{}.png'.format(j, k))


if __name__ == "__main__":
	
	import csv
	from mpi4py import MPI as MPI4py
	import sys
	import shutil

	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)

	world = MPI4py.COMM_WORLD
	world_size = world.Get_size()
	world_rank = world.Get_rank()

	block_size_layer_1 = 24
	block_size_layer_2 = 12
	block_size_layer_ts = 24
	TSN = 125

	# Define paths 
	# Run script in home folder
	# job_path: all intermediate grids and policy functions 
	# 	saved here
	# scr_path: parameter draws and final policies saved here
	# settings_path: settings folder relative to project root
	#job_path = get_jobfs_path()
	job_path = "/scratch/pv33/ls_model_temp2"
	scr_path = "/scratch/pv33/ls_model_temp2"
	settings_path = "settings"
	home_path = "/home/141/as3442/Eggsandbasket2.0/eggsandbaskets/"
	

	# Read settings
	eggbasket_config, param_random_bounds = read_settings(settings_path)

	gender = sys.argv[1]
	#gender = 'male'

	if gender == 'male':
	# Solve male baseline (Initial submit version may 12,2021)
	# Note '9GLB6E_20210501-181303_baseline_male' is used as male baseline ID
		model_name = 'baseline_male_v9'
		params = eggbasket_config['male']
		top_id = pickle.load(open("/scratch/pv33/ls_model_temp2/baseline_male_v9/topid.smms","rb"))
		params = pickle.load(open("/scratch/pv33/ls_model_temp2/baseline_male_v9/params/{}_params.pkl".format(top_id),"rb"))
		param_dict = eggbasket_config['male']
		param_dict['parameters'] = params
		#sampmom = pickle.load(open("/scratch/pv33/ls_model_temp2/baseline_male/latest_sampmom.smms".format(model_name),"rb"))
		sampmom = [0,0]

		#param_dict['rho'] = 0.05
		#param_dict['grid_size_H'] = 20
		#param_dict['gamma'] = 4
		#shutil.rmtree(job_path+'/{}/'.format(model_name))
		
	# Solve female baseline

	if gender == 'female':
		model_name = 'baseline_female'
		params = eggbasket_config['female']
		#top_id = pickle.load(open("/scratch/pv33/ls_model_temp/final_female_v3/topid.smms".format(model_name),"rb"))
		sampmom = pickle.load(open("/scratch/pv33/ls_model_temp/final_male_v3/latest_sampmom.smms".format(model_name),"rb"))
		#param_dict = pickle.load(open("/scratch/pv33/ls_model_temp/final_female_v3/{}_acc_0/params.smms".format(top_id),"rb")) 
		params['parameters'] = param_dict

	layer_1_comm, layer_2_comm, layer_ts_comm, node_world,\
			cross_node_world, cross_layer1_world = gen_communicators(world,\
													block_size_layer_1,\
													block_size_layer_2,\
													block_size_layer_ts)
	#if node_world.rank == 0:  
	#	shutil.rmtree(job_path+'/{}/'.format(model_name))
	print(cross_layer1_world.rank)
	if layer_1_comm.rank == 0: 
		#print(param_dict['parameters']['A_max_WE'])
		cov_mat = np.zeros(np.shape(sampmom[1]))
		LS_models =  lifecycle_model\
						.LifeCycleParams(model_name, param_dict,scr_path,
						  random_draw = False, 
						  random_bounds = param_random_bounds, # parameter bounds for randomly generated params
						  param_random_means = sampmom[0], # mean of random param distribution 
						  param_random_cov = cov_mat, 
						  uniform = False)
	else:
		LS_models = None
		test = None

	LS_models = layer_1_comm.bcast(LS_models, root = 0)

	print("Rank {} on world is rank {} on layer 1 and rank {} on layer 2 and rank {} on cross_layer1_world model ID is {}"\
			.format(world_rank,layer_1_comm.rank, layer_2_comm.rank, cross_layer1_world.rank, LS_models.param_id))

	pickle.dump(LS_models.param_id,open(scr_path + "/{}/single_ID_latest.smms".format(model_name),"wb") )

	if layer_1_comm.rank < layer_2_comm.size:
		og  = LS_models.og_DB
		og.ID = 'VN7R6M_20211009-123010_baseline_male_v8'

	else:
		og  = LS_models.og_DC
		og.ID = 'VN7R6M_20211009-123010_baseline_male_v8'

	if layer_1_comm.rank == 0:
		param_id = LS_models.param_id
	else: 
		param_id = None

	param_id_list = world.gather(param_id, root = 0)

	if world.rank == 0:
		param_id_list = print([item for item in param_id_list if item is not None])

	#from solve_policies.retiree_solver import retiree_func_factory
	#gen_R_pol = retiree_func_factory(og)
	#ret_pols = gen_R_pol(layer_1_comm, noplot= False)

	#if layer_1_comm.rank ==0:
	#	plot_retiree(ret_pols, og)
	start = time.time()
	policies = generate_worker_pols(og,world,layer_2_comm, load_retiree = False, jobfs_path = job_path, verbose = True)

	node_world.Barrier()
	
	# Generate moments 
	
	if cross_layer1_world.rank == 0:
		
		if layer_1_comm.rank == 0:

			U = np.random.rand(6,100,TSN,100) 

			gen_panel_ts('male',og,U, TSN,job_path)
			generate_TSDF, load_pol_array = genprofiles_operator(og)
			mod_name = og.mod_name
			print(time.time()- start)
			TSALL_10_df = pd.read_pickle(job_path + '/{}/TSALL_10_df.pkl'.format(og.mod_name +'/'+ og.ID + '_acc_0'))  
			TSALL_14_df	= pd.read_pickle(job_path + '/{}/TSALL_14_df.pkl'.format(og.mod_name +'/'+ og.ID + '_acc_0')) 
			
			#TSALL_10_df,TSALL_14_df  = gen_panel_ts()
			moments_male = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_male') 

			moments_female = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_female')

			moments_sim_sorted = sortmoments(moments_male,\
												 moments_female)

			os.chdir(os.path.expanduser("~/Eggsandbasket2.0/eggsandbaskets/plots")) 
			moments_sim_sorted.to_csv("moments_sorted_test.csv")

			moments_data = pd.read_csv('{}/moments_data_plot.csv'\
							.format(home_path + settings_path))

			plot_moments_ts(moments_male,moments_female, moments_data, home_path, model_name)

	node_world.Barrier()

	if cross_layer1_world.rank == 1:

		if layer_1_comm.rank == 0:

			U = np.random.rand(6,100,TSN,100) 

			gen_panel_ts('male',og,U, TSN,job_path)

			generate_TSDF, load_pol_array = genprofiles_operator(og)


			mod_name = og.mod_name
			print(time.time()- start)
			TSALL_10_df = pd.read_pickle(job_path + '/{}/TSALL_10_df.pkl'.format(og.mod_name +'/'+ og.ID + '_acc_0'))  
			TSALL_14_df	= pd.read_pickle(job_path + '/{}/TSALL_14_df.pkl'.format(og.mod_name +'/'+ og.ID + '_acc_0')) 
			
			moments_male = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_male') 

			moments_female = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_female')


			moments_sim_sorted = sortmoments(moments_male,\
												 moments_female)
			os.chdir(os.path.expanduser("~/Eggsandbasket2.0/eggsandbaskets/plots")) 
			moments_sim_sorted.to_csv("moments_sorted_test.csv")



