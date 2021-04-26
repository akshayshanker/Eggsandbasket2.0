import yaml
import dill as pickle
import glob 
import copy
import os
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np

# Housing model modules
import lifecycle_model 
from solve_policies.worker_solver import generate_worker_pols
from generate_timeseries.tseries_generator import gen_panel_ts,\
					 gen_moments, sortmoments, genprofiles_operator


def gen_communicators(world, block_size_layer_1, 
					block_size_layer_2,
					block_size_layer_ts):

	# Number cores to process each LS model 
	# Should be 2x2
	blocks_layer_1 = world_size/block_size_layer_1
	color_layer_1 = int(world_rank/block_size_layer_1)
	key_layer1 = int(world_rank%block_size_layer_1)

	layer_1_comm = world.Split(color_layer_1,key_layer1)

	layer_1_rank = layer_1_comm.Get_rank()

	# Number of cores to process each DB/DC
	blocks_layer_2 = block_size_layer_1/block_size_layer_2
	color_layer_2 = int(layer_1_rank/block_size_layer_2)
	key_layer2 = int(layer_1_rank%block_size_layer_1)

	layer_2_comm = layer_1_comm.Split(color_layer_2,key_layer2)

	layer_2_rank = layer_2_comm.Get_rank()

	# TS generation layers
	
	blocks_layer_ts = world_size/block_size_layer_ts
	color_layer_ts = int(world_rank/block_size_layer_ts)
	key_layer_ts = int(world_rank%block_size_layer_ts)

	layer_ts_comm = world.Split(color_layer_ts,key_layer_ts)

	return layer_1_comm, layer_2_comm, layer_ts_comm

if __name__ == "__main__":
	import csv
	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)

	world = MPI4py.COMM_WORLD
	world_size = world.Get_size()
	world_rank = world.Get_rank()

	block_size_layer_1 = 8
	block_size_layer_2 = 4
	block_size_layer_ts = 8

	# Read settings
	with open("settings/settings.yml", "r") as stream:
		eggbasket_config = yaml.safe_load(stream)

	param_random_bounds = {}
	with open('settings/random_param_bounds.csv', newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],\
				row['UB']])

	model_name = 'final_male_v2'
	top_id = pickle.load(open("/scratch/pv33/ls_model_temp/{}/topid.smms".format(model_name),"rb"))
	#top_id = 'AMWGWF_20210415-132028_test_3'
	sampmom = pickle.load(open("/scratch/pv33/ls_model_temp/final_male_v2/latest_sampmom.smms","rb"))
	#sampmom[0][9]= 2
	#sampmom[0][2]= 250
	#sampmom[0][21]= 1
	#param_dict = pickle.load(open("/scratch/pv33/ls_model_temp/{}/{}_acc_0/params.smms".format(model_name, top_id),"rb")) 
	param_dict = eggbasket_config['male']

	layer_1_comm, layer_2_comm, layer_ts_comm = gen_communicators(world,\
													block_size_layer_1,\
													block_size_layer_2,\
													block_size_layer_ts)
	if layer_1_comm.rank == 0: 
		cov_mat = np.zeros(np.shape(sampmom[1]))
		LS_models =  lifecycle_model\
						.LifeCycleParams('final_male_v2', param_dict, random_draw = True, 
		                  random_bounds = param_random_bounds, # parameter bounds for randomly generated params
		                  param_random_means = sampmom[0], # mean of random param distribution 
		                  param_random_cov = cov_mat, 
		                  uniform = False)
		test = 0
		print('test')
		LS_models.og_DB.ID = top_id
		LS_models.og_DC.ID = top_id
	else:
		LS_models = None
		test = None

	LS_models = layer_1_comm.bcast(LS_models, root = 0)

	print("Rank {} on world is rank {} on layer 1 and rank {} on layer 2 and model ID is {}"\
			.format(world_rank,layer_1_comm.rank, layer_2_comm.rank,LS_models.param_id))

	pickle.dump(LS_models.param_id,open("/scratch/pv33/ls_model_temp/final_male_v2/single_ID.smms","wb") )

	if layer_1_comm.rank == 0 or layer_1_comm.rank == 1 or layer_1_comm.rank == 2 or layer_1_comm.rank == 3 :
		og  = LS_models.og_DB

	if layer_1_comm.rank == 4 or layer_1_comm.rank == 5 or layer_1_comm.rank == 6 or layer_1_comm.rank == 7 :
		og  = LS_models.og_DC

	if layer_1_comm.rank == 0:
		param_id = LS_models.param_id
	else: 
		param_id = None

	param_id_list = world.gather(param_id, root = 0)

	if world.rank == 0:
		param_id_list = print([item for item in param_id_list if item is not None])

	policies = generate_worker_pols(og,world,layer_2_comm, load_retiree = 1, gen_newpoints = False)

	# Generate moments 
	if layer_1_comm.rank == 0:
		
		ID = param_id
		TSN = 100
		U = np.random.rand(6,100,TSN,100) 

		TSALL_10_df, TSALL_14_df = gen_panel_ts(og,U, TSN)

		moments_male = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_male') 

		moments_female = gen_moments(copy.copy(TSALL_10_df), copy.copy(TSALL_14_df)).add_suffix('_female')


		moments_sim_sorted = sortmoments(moments_male,\
		                                     moments_female)

		pickle.dump(moments_male, open('/scratch/pv33/ls_model_temp/{}/moments_male'.format(LS_models.mod_name), "wb"))



