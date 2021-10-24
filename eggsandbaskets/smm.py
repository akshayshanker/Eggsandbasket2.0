  
"""
Module estimates HousingModel using Simualted Method of Moments
Minimisation performed using Cross-Entropy method (see Kroese et al)

Script must be run using Mpi with cpus = 24 * number
of cross entropy draws. (Number of cores per draw: communicator topolgoy)


Example (on Gadi normal compute node):

module load python3/3.7.4
module load openmpi/4.0.5

 
mpiexec -n 480 python3 -m mpi4py smm.py

"""

# Import packages
import yaml
import gc
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import csv
import time
import dill as pickle 
import copy
import sys
import pandas as pd
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression
import shutil

# Housing model modules
import lifecycle_model 
from solve_policies.worker_solver import generate_worker_pols
from generate_timeseries.tseries_generator import gen_panel_ts,\
					 gen_moments, genprofiles_operator
from generate_timeseries.ts_helper_funcs import sortmoments




def reset_smm(mod_name, scr_path):

	mod_name  = mod_name
	TSN = 500
	d = 3
	path = scr_path + "/" + "{}".format(mod_name)
	os.makedirs(path, exist_ok = True) 

	t = 0

	pickle.dump(t, open(path + "/t.smms","wb")) 
	sampmom = [0,0]
	pickle.dump(sampmom,open(path + "/latest_sampmom.smms","wb"))


	S_star,gamma_XEM	= np.full(d,0), np.full(d,0)


	U = np.random.rand(6,100,TSN,100)   

	pickle.dump(U,open(path + "/seed_U.smms","wb"))

	pickle.dump(gamma_XEM,open(path + "/gamma_XEM.smms","wb"))

	pickle.dump(S_star,open(path + "/S_star.smms","wb"))

	pickle.dump(sampmom,open(path + "/latest_sampmom.smms","wb"))

	return None

def gen_communicators(big_world,
						block_size_layer_1, 
						block_size_layer_2,
						block_size_layer_ts):

	""" Generates sub-communicators. There are N 
		Layer 1 communicator groups with 8 cores each,
		each solving a model for a parameter vector. There are N/2 
		Layer 2 communicator groups with 4 cores each, each layer 2
		group solving a DB or DC for a parameter vector""" 

	# Number cores to process each LS model 
	# Should be 2x2

	# create worlds on each node so they can share mem
	# we
	node_world = big_world.Split_type(split_type = 0)
	big_world_rank = big_world.Get_rank()

	world_size = node_world.Get_size()
	world_rank = node_world.Get_rank()
	blocks_layer_1 = world_size/block_size_layer_1
	color_layer_1 = int(world_rank/block_size_layer_1)
	key_layer1 = int(world_rank%block_size_layer_1)

	layer_1_comm = node_world.Split(color_layer_1,key_layer1)
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

	layer_ts_comm = node_world.Split(color_layer_ts,key_layer_ts)

	cross_node_world = big_world.Split(int(world_rank),big_world.Get_rank())

	# Generate cross- layer 1 communicator from node world
	# Each rank in this communicator belongs to a distinct layer 1 communicator 

	cross_layer1_world = node_world\
				.Split(int(layer_1_rank),node_world.Get_rank())

	return layer_1_comm, layer_2_comm, layer_ts_comm, node_world,\
			cross_node_world, cross_layer1_world

def gen_format_moments(TS1,TS2, moments_data, moments_weights, gender):
	"""Gen simulated moments, labels
		and sorts simulated and data mooments
		and generates numpy arrays"""

	moments_male = gen_moments(copy.copy(TS1),copy.copy(TS2))   
	moments_female = gen_moments(copy.copy(TS1),copy.copy(TS2)) 
	moments_female = moments_female.add_suffix('_female')
	moments_male = moments_male.add_suffix('_male')
	moments_sim_sorted = sortmoments(moments_male,\
										 moments_female)

	#plot_moments_ts(moments_male,moments_female, moments_data, home_path, model_name)

	moments_sim_sorted = pd.concat([moments_male["Age_wave10_male"]\
								.reset_index().iloc[:,1],\
								moments_sim_sorted],\
								axis =1)  
	moments_sim_sorted = moments_sim_sorted.rename(columns =\
							 {'Age_wave10_male':'Age_wave10'})

	moments_data.columns = moments_sim_sorted.columns


	moments_weights.columns = moments_sim_sorted.columns
	
	moments_sim_sorted = moments_sim_sorted\
				.loc[:,moments_sim_sorted.columns.str.endswith('_'+gender)] 
	
	moments_sim_array = np.array(np.ravel(moments_sim_sorted))

	#moments_sim_array[np.isnan(moments_sim_array)] = 0

	moments_data = moments_data.loc[:,moments_data.columns.str\
									.endswith('_'+gender)] 
	moments_weights = moments_weights.loc[:,moments_weights.columns.str\
									.endswith('_'+gender)] 

	moments_data_array = np.array(np.ravel(moments_data))
	moments_weights_array = np.array(np.ravel(moments_weights))

	return moments_sim_array, moments_data_array, moments_weights_array

def gen_RMS(t, LS_models, gender, moments_data, moments_weights, world_comm, layer_1_comm,\
				layer_2_comm,node_world, cross_node_world, \
				cross_layer1_world, TSN,U, job_path, scr_path):
	"""
	Simulate model for param vector and generate root mean square error 
	between simulated moments for HousingModel 
	and data moments 
	"""
	# In each layer 1 group, first two ranks 
	# perform DB and second two ranks 
	# perform DC 
	if layer_1_comm.rank < layer_2_comm.size:
		og  = LS_models.og_DB

	else:
		og  = LS_models.og_DC
	
	# Each layer two group solves the model  
	if world_comm.rank == 0: 
		print("Solving model.")
	
	policies = generate_worker_pols(og,
									world_comm,
									layer_2_comm,
									load_retiree = 0,
									jobfs_path = job_path, verbose = False)

	layer_1_comm.Barrier()

	if world_comm.rank == 0:
		print("Solved model.")
	else:
		pass 
	
	node_world.Barrier()

	if cross_layer1_world.rank == 0:
		if layer_1_comm.rank == 0:
			#TSALL_10_df, TSALL_14_df = gen_panel_ts(gender,og,U, TSN, job_path)
			#Process = pathos.helpers.mp.Process
			#print(job_path)
			#p = Process(target = gen_panel_ts, args = (gender,og,U, TSN,job_path, ))
			#p.start()
			#p.join() 
			gen_panel_ts(gender,og,U, TSN,job_path)
			TSALL_10_df = pd.read_pickle(job_path + '/{}/TSALL_10_df.pkl'\
								.format(og.mod_name +'/'+ og.ID + '_acc_0')) 
			TSALL_14_df	= pd.read_pickle(job_path + '/{}/TSALL_14_df.pkl'\
								.format(og.mod_name +'/'+ og.ID + '_acc_0')) 
			#del p
			gc.collect()

	else:
		pass 

	node_world.Barrier()

	# Process time-series only one at a time on each node
	if cross_layer1_world.rank == 1:
		if layer_1_comm.rank == 0:
			#TSALL_10_df, TSALL_14_df = gen_panel_ts(gender,og,U, TSN, job_path)
			#Process = pathos.helpers.mp.Process
			#p = Process(target = gen_panel_ts, args = (gender,og,U, TSN,job_path, ))
			#p.start()
			#p.join() 
			gen_panel_ts(gender,og,U, TSN,job_path)
			TSALL_10_df = pd.read_pickle(job_path + '/{}/TSALL_10_df.pkl'\
							.format(og.mod_name +'/'+ og.ID + '_acc_0'))
			TSALL_14_df	= pd.read_pickle(job_path + '/{}/TSALL_14_df.pkl'\
							.format(og.mod_name +'/'+ og.ID + '_acc_0')) 
			#del p
			gc.collect()
	else:
		pass 

	if layer_1_comm.rank == 0: 

		# We  use the LifeCycle model instance on layer 1 master
		# to generate the TS since it contains the parameter ID (og.ID)
		#plot_path = scr_path + '/' + og.mod_name +'/'+ og.ID + '/'

		moments_sim_array, moments_data_array, moments_weights_array \
		= gen_format_moments(TSALL_10_df, TSALL_14_df, moments_data, moments_weights, gender)


		del TSALL_10_df
		del TSALL_14_df 
		gc.collect()

		moments_data_nonan \
					= moments_data_array[np.where(~np.isnan(moments_data_array))]
		moments_weights_nonan = moments_weights_array[np.where(~np.isnan(moments_data_array))]
		moments_sim_nonan \
					= moments_sim_array[np.where(~np.isnan(moments_data_array))]
		demon_mom = np.abs(moments_data_nonan)
		deviation_r = np.abs((moments_sim_nonan - moments_data_nonan)/demon_mom)
		deviation_d = np.abs(moments_data_nonan-moments_sim_nonan)

		deviation_r[np.where(np.abs(moments_data_nonan)<1)]\
				 = deviation_d[np.where(np.abs(moments_data_nonan)<1)]
		
		N_err = len(deviation_r)
		deviation_r = deviation_r
		deviation_r[moments_sim_nonan == 0] = 1E50
		deviation_r[moments_sim_nonan == 1] = 1E50
		deviation_r[np.where(np.isnan(moments_sim_nonan))] = 1E50
		

		return 1-(1/N_err)*np.sqrt(np.sum(np.square(deviation_r)))

	else:
		return None

def load_tm1_iter(model_name, scr_path):
	""" Initializes array of best performer and least best in the 
		elite set (see notationin Kroese et al)
	""" 

	S_star = pickle\
				.load(open(scr_path + "/{}/S_star.smms".format(model_name),"rb"))
	gamma_XEM = pickle\
				.load(open(scr_path + "/{}/gamma_XEM.smms".format(model_name),"rb"))
	t = pickle.load(open(scr_path + "/{}/t.smms"\
				.format(model_name),"rb"))
	sampmom = pickle.load(open(scr_path+ "/{}/latest_sampmom.smms"\
				.format(model_name),"rb"))

	return gamma_XEM, S_star,t, sampmom

def iter_SMM(eggbasket_config, 
							 model_name,
							 gender,
							 param_random_bounds, 
							 sampmom, 	   # t-1 parameter means 
							 moments_data, # data moments 
							 moments_weights,
							 world_comm,
							 layer_1_comm, # layer 1 communicator 
							 layer_2_comm, # layer 2 communicator 
							 node_world, 
							 cross_node_world,
							 cross_layer1_world, 
							 TSN,
							 U, 		# fixed and stored model errors 
							 gamma_XEM, # lower elite performer
							 S_star, 	# upper elite performer
							 t, job_path,
							 scr_path,
							 reset_draws): 		# iteration number 
	
	""" Initializes parameters and LifeCycel model and peforms 
		one iteration of the SMM, returning updated sampling distribution

	'"""
	# Generate LifeCycleParam class (master on layer 1: new random sample)
	indexed_errors = None
	parameter_list = None
	if layer_1_comm.rank == 0: 
		if t == 0:
			uniform = True

		else:
			uniform = False

		LS_models = lifecycle_model.LifeCycleParams(model_name,
									eggbasket_config[gender],
									scr_path,
									random_draw = True,
									random_bounds = param_random_bounds, 
									param_random_means = sampmom[0], 
									param_random_cov = sampmom[1], 
									uniform = uniform)
		parameters = LS_models.parameters
	else:
		LS_models  = None
		parameters = None
	
	# Broadcast LifeCycleParam from each Layer 1 master to workers
	LS_models = layer_1_comm.bcast(LS_models, root = 0)
	
	if world_comm.rank == 0:
		print("Random Parameters drawn, distributng iteration {}".format(t))
	else:
		pass 
		
	def SMM_objective():
		"""SMM objective to be maximised 
		as function of params"""

		RMS =  gen_RMS(t, LS_models, gender, 
						moments_data,\
						moments_weights, 
						world_comm,
						layer_1_comm,
						layer_2_comm,\
						node_world,\
						cross_node_world,\
						cross_layer1_world,\
						TSN,\
						U, job_path, scr_path)
		if layer_1_comm.rank == 0:
			return [LS_models.param_id, np.float64(RMS)]
		else: 
			return ['none', 0]

	errors_ind = SMM_objective()

	if layer_1_comm.rank == 0:
		Path(scr_path + "/" + model_name + "/errs/")\
					.mkdir(parents = True, exist_ok = True)
		Path(scr_path + "/" + model_name + "/params/")\
					.mkdir(parents = True, exist_ok = True)
		pickle.dump(errors_ind, open(scr_path + "/" + model_name\
					 + "/errs/{}_errors.pkl".format(LS_models.param_id), "wb"))
		pickle.dump(parameters, open(scr_path + "/" + model_name\
					 + "/params/{}_params.pkl".format(LS_models.param_id), "wb"))

	# Gather parameters and corresponding errors from all ranks
	# Only layer 1 rank 0 values are not None
	layer_1_comm.Barrier()

	del LS_models
	gc.collect()

	if node_world.rank == 0:
		len_dir = 0
		
		while len_dir < N_elite + 5:

			dir_err = os.listdir(scr_path + "/" + model_name + "/errs/")
			time.sleep(5)
			dir_params = os.listdir(scr_path + "/" + model_name + "/params/")
			indexed_errors = []
			parameter_list = []
			time.sleep(5)
			len_dir = len(dir_err)

		for file in dir_err:
			errors = pickle.load(open(scr_path + "/" + model_name\
								 + "/errs/" + file, "rb"))
			indexed_errors.append(errors)	

		for file in dir_params:
			params_list = pickle.load(open(scr_path + "/" + model_name\
								 + "/params/" + file, "rb"))
			parameter_list.append(params_list)


		indexed_errors = np.array(indexed_errors)

		parameter_list_dict = dict([(param['param_id'], param)\
							 for param in parameter_list])
		
		errors_arr = np.array(indexed_errors[:,1]).astype(np.float64)

		error_indices_sorted = np.take(indexed_errors[:,0],\
											np.argsort(-errors_arr))
		errors_arr_sorted = np.take(errors_arr,\
											np.argsort(-errors_arr))
		
		number_N = len(error_indices_sorted)
		
		elite_errors = errors_arr_sorted[0: N_elite]
		elite_indices = error_indices_sorted[0: N_elite]

		weights = np.exp((elite_errors - np.min(elite_errors))\
						/ (np.max(elite_errors)\
							- np.min(elite_errors)))
		gamma_XEM = np.append(gamma_XEM,elite_errors[-1])
		S_star = np.append(S_star,elite_errors[0])

		error_gamma = gamma_XEM[d +t] \
						- gamma_XEM[d +t -1]
		error_S = S_star[int(d +t)]\
						- S_star[int(d +t -1)]

		means, cov = gen_param_moments(elite_errors,sampmom,\
										parameter_list_dict,\
										param_random_bounds,\
										elite_indices, weights,t)
		
		if world.rank == 0:
			print("...generated and saved sampling moments")
		else:
			pass 

	node_world.Barrier()
	if node_world.rank == 0:
		shutil.rmtree(job_path+'/{}/'.format(model_name))
	else:
		pass 

	if node_world.rank == 0:
		return number_N, [means, cov], gamma_XEM, S_star,\
					error_gamma, error_S, elite_indices[0]
	else:
		return None


def gen_param_moments(elite_errors,\
						sampmom,\
						parameter_list_dict,
						param_random_bounds,
						selected,
						weights,
						t,
						rho_smooth = 0,
						rho_gd = 0):

	""" Estimate params of a sampling distribution

	Parameters
	----------
	parameter_list_dict: Dict
						  Dictionary with all paramameters
						  with ID keys
	selected: 2D-array
						   set of elite paramters IDs and errors

	Returns
	-------

	means

	cov
	"""

	sample_params = []

	for i in range(len(selected)):
		rand_params_i = []
		for key in param_random_bounds.keys():
			rand_params_i.append(\
				float(parameter_list_dict[selected[i]][key]))
		
		sample_params.append(rand_params_i)

	# Get list of bounds
	random_param_bounds_ar = np.array([(bdns) for key, bdns \
										in param_random_bounds.items()] ) 

	# Evaluate gradients
	sample_params_reg = np.array(sample_params)
	reg = LinearRegression().fit(sample_params_reg, elite_errors)
	coeffs = reg.coef_

	sample_params = np.array(sample_params)
	means = np.average(sample_params, axis=0)
	cov = np.cov(sample_params, rowvar=0)

	grad_new_params_1 = means + coeffs
	grad_new_params = np.clip(grad_new_params_1,\
							 random_param_bounds_ar[:, 1],\
							 random_param_bounds_ar[:, 0])

	if t>0:

		means = sampmom[0]*rho_smooth + (1-rho_smooth)*means
		cov = sampmom[1]*rho_smooth + (1-rho_smooth)*cov
	else: 

		pass 

	means = (1-rho_gd) * means + rho_gd* grad_new_params

	return means, cov

if __name__ == "__main__":

	from mpi4py import MPI as MPI4py
	import sys
	from util.helper_funcs import get_jobfs_path, read_settings
	import os
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)

	# MPI commincator topology 
	world = MPI4py.COMM_WORLD
	block_size_layer_1 = 24
	block_size_layer_2 = 12
	block_size_layer_ts = 24

	# Estimation parameters  
	tol = 1E-1
	TSN = 250
	N_elite = 35
	d = 3
	number_N = None
	error = 1

	# Read in gender, model name and reset
	gender = sys.argv[1]
	model_name = sys.argv[2]
	reset_draws = sys.argv[3]
	reset_init = sys.argv[4]

	settings_folder = 'settings'
	job_path = get_jobfs_path()
	scr_path = "/scratch/pv33/ls_model_temp2"
	home_path = "/home/141/as3442/Eggsandbasket2.0/eggsandbaskets/"

	# If starting a fresh estimation, clear scratch directories 
	if reset_draws == 'True' and world.rank == 0:
		print("Clearing folders...")
		
		shutil.rmtree(scr_path + "/" + model_name + "/errs/",\
						 ignore_errors = True)
		shutil.rmtree(scr_path + "/" + model_name + "/params/",\
						 ignore_errors = True)

		Path(scr_path + "/" + model_name + "/errs/")\
					.mkdir(parents = True, exist_ok = True)
		Path(scr_path + "/" + model_name + "/params/")\
					.mkdir(parents = True, exist_ok = True)
		
		if reset_init == 'True':
			reset_smm(model_name, scr_path)
		
	world.Barrier()
	gamma_XEM, S_star,t, sampmom = load_tm1_iter(model_name,scr_path)

	# Generate communicators 
	if world.rank == 0:
		print("Initializing communicators...")
	
	# Create communicators 
	layer_1_comm, layer_2_comm, layer_ts_comm, node_world, cross_node_world,\
	cross_layer1_world = gen_communicators(world,
											block_size_layer_1,
											block_size_layer_2,
											block_size_layer_ts)

	eggbasket_config, param_random_bounds,\
		 param_random_bounds_big = read_settings(settings_folder)

	if gender == 'male_big' or gender == 'female_big':
		param_random_bounds = param_random_bounds_big


	world.Barrier()

	# Load data moments and shocks 
	if layer_1_comm.rank == 0:
		moments_data = pd.read_csv('{}/moments_data.csv'\
							.format(settings_folder))
		moments_weights = pd.read_csv('{}/moments_weights.csv'\
							.format(settings_folder))
		
		U = pickle.load(open(scr_path + "/{}/seed_U.smms"\
					.format(model_name),"rb"))
	else:
		moments_data = None
		U = None
		moments_weights = None

	# Iterate on cross-entropy loop 
	while t< 50 and error > tol:
		start = time.time()	
		os.chdir(home_path)
		iter_return = iter_SMM(eggbasket_config,
								 model_name,
								 gender,
								 param_random_bounds,
								 sampmom,
								 moments_data,
								 moments_weights,
								 world, 
								 layer_1_comm,
								 layer_2_comm,
								 node_world,
								 cross_node_world, 
								 cross_layer1_world, 
								 TSN,
								 U,
								 gamma_XEM,
								 S_star,
								 t,
								 job_path,
								 scr_path,
								 reset_draws) 
		
		if node_world.rank == 0:
			number_N, sampmom, gamma_XEM, S_star,\
				error_gamma, error_S, top_ID = iter_return
			error = np.abs(np.max(sampmom[1]))

		sampmom = node_world.bcast(sampmom, root = 0)
		number_N = node_world.bcast(number_N, root = 0)
		gamma_XEM = node_world.bcast(gamma_XEM, root = 0)
		S_star = node_world.bcast(S_star, root = 0)
		error = node_world.bcast(error, root= 0)

		if world.rank == 0:

			pickle.dump(gamma_XEM,open(scr_path + "/{}/gamma_XEM.smms"\
						.format(model_name),"wb"))
			pickle.dump(S_star,open(scr_path + "/{}/S_star.smms"\
						.format(model_name),"wb"))
			pickle.dump(t,open(scr_path + "/{}/t.smms"\
						.format(model_name),"wb"))
			pickle.dump(sampmom,open(scr_path + "/{}/latest_sampmom.smms"\
						.format(model_name),"wb"))
			pickle.dump(top_ID, open(scr_path + "/{}/topid.smms"\
						.format(model_name),"wb"))
			
			print("Iteration no. {} in {} min. on {} samples,\
						 elite gamma error: {} and elite S error: {}"\
						.format(t, (time.time()-start)/60,\
								 number_N, error_gamma, error_S))
			print("....cov error: {}."\
					.format(np.abs(np.max(sampmom[1]))))
		else:
			pass 

		gc.collect()
		node_world.Barrier()
		t = t+1