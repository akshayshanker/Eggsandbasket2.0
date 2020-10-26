  
"""
Module estiamates HousingModel using Simualted Method of Moments
Minimisation performed using Cross-Entropy method (see Kroese et al)

Script must be run using Mpi: 

Example (on Gadi):

module load python3/3.7.4
module load openmpi/4.0.2

alias mpython='mpiexec -np 480 `which python3`'
 
mpython SMM.py

"""

# import packages
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')


from collections import defaultdict
from numpy import genfromtxt
import csv
import time
import dill as pickle 
from randparam   import rand_p_generator
import copy
import sys
import pandas as pd

# Housing model modules
import lifecycle_model 
from solve_policies.worker_solver import generate_worker_pols
from generate_timeseries.tseries_generator import gen_panel_ts, gen_moments, sortmoments, genprofiles_operator


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


def gen_format_moments(TS1,TS2, moments_data):
	"""Gen simulated moments, labels
		and sorts simulated and data mooments
		and generates numpy arrays"""

	moments_male        	= gen_moments(copy.copy(TS1),copy.copy(TS2))   
	moments_female      	= gen_moments(copy.copy(TS1),copy.copy(TS2)) 
	moments_female      	= moments_female.add_suffix('_female')
	moments_male          	= moments_male.add_suffix('_male')
	moments_sim_sorted   	= sortmoments(moments_male,\
										 moments_female)

	moments_sim_sorted 		= pd.concat([moments_male["Age_wave10_male"]\
								.reset_index().iloc[:,1],\
								moments_sim_sorted],\
								axis =1)  
	moments_sim_sorted 		= moments_sim_sorted.rename(columns =\
							 {'Age_wave10_male':'Age_wave10'})

	moments_data.columns 	= moments_sim_sorted.columns

	
	moments_sim_sorted =\
		moments_sim_sorted\
		.loc[:,moments_sim_sorted.columns.str.endswith('_male')] 
	
	moments_sim_array  = np.array(np.ravel(moments_sim_sorted))

	moments_sim_array[np.isnan(moments_sim_array)] = 0

	moments_data =\
	moments_data.loc[:,moments_data.columns.str.endswith('_male')] 

	moments_data_array   = np.array(np.ravel(moments_data))

	return moments_sim_array, moments_data_array

def gen_RMS(LS_models, moments_data, layer_1_comm,layer_2_comm, TSN,U):
	"""
	Generate root mean square error 
	between simulated moments for HousingModel 
	and data moments 

	"""

	# In each layer 1 group, first two ranks 
	# perform DB and second two ranks 
	# perform DC 
	if layer_1_comm.rank == 0 or layer_1_comm.rank == 1:
		og  = LS_models.og_DB

	if layer_1_comm.rank == 2 or layer_1_comm.rank == 3:
		og  = LS_models.og_DC
	
	# Each layer two group solves the model  
	policies = generate_worker_pols(og,layer_2_comm, load_retiree = 1,\
										 gen_newpoints = False)

	# Generate time series on each Layer 1 master 
	if layer_1_comm.rank == 0: 

		# We  use the LifeCycle model instance on layer 1 master
		# to generate the TS since it contains the parameter ID (og.ID)
		generate_TSDF = genprofiles_operator(og)
		TSALL_10_df, TSALL_14_df = gen_panel_ts(og,U, TSN)

		# Generate and sort moments
		moments_sim_array, moments_data_array \
		= gen_format_moments(TSALL_10_df, TSALL_14_df, moments_data)

		del TSALL_10_df
		del TSALL_14_df 
		gc.collect()

		deviation = (moments_sim_array\
									[~np.isnan(moments_data_array)]\
									  - moments_data_array\
									  [~np.isnan(moments_data_array)])
		norm  = np.sum(np.square(moments_data_array[~np.isnan(moments_data_array)]))
		N_err = len(deviation)

		return 1-np.sqrt((1/N_err)*np.sum(np.square(deviation))/norm)
	# Return none if not layer 1 master 
	else
		return None

	def load_tm1_iter(model_name):
		gamma_XEM = pickle.load(open("/scratch/pv33/{}/gamma_XEM.smms".format(model_name),"rb"))
		S_star = pickle.load(open("/scratch/pv33/{}/S_star.smms".format(model_name),"rb"))
		t = pickle.load(open("/scratch/pv33/{}/t.smms".format(model_name),"rb"))
		sampmom = pickle.load(open("/scratch/pv33/Archive/{}/latest_means_iter.smms".format(model_name),"rb"))
	return gamma_XEM, S_star,t, sampmom




def iter_SMM(eggbasket_config, 
			 sampmom, # t-1 parameter means 
			 moments_data, # data moments 
			 layer_1_comm, 
			 layer_2_comm,
			 TSN,
			 U, # fixed and stored model errors 
			 gamma_XEM, # lower error array
			 S_star # upper error array
			):
	""" iterate on the SMM cross-entropy
	'"""

	# Generate LifeCycleParam class (each master on layer 1 generates a random sample)
	indexed_errors = None
	parameter_list = None
	if layer_1_comm.rank == 0: 
		LS_models =  lifecycle_model\
						.LifeCycleParams(model_name,\
									eggbasket_config['baseline_lite'],
									random_draw = True,
									random_bounds = param_random_bounds, 
									param_random_means = sampmom[0], 
									param_random_cov = sampmom[1], 
									uniform = False)
		parameters = LS_models.paramters
	else:
		LS_models = None
		parameters = None
	
	# Broadcast LifeCycleParam from each Layer 1 master to workers
	LS_models = layer_1_comm.bcast(LS_models, root = 0)
	
	if world.rank ==0:
		print("Distributng iter {}".format(t))
		errors_ind = [0,0]

	def SMM_objective():
		"""SMM objective to be maximised 
		as function of params"""

		RMS =  gen_RMS(LS_models,
						moments_data,\
						layer_1_comm,
						layer_2_comm,\
						TSN,\
						U)
		if layer_1_comm.rank == 0:
			return [LS_models.ID, RMS]
		else 
			return None

	errors_ind = SMM_objective()

	comm.Barrier()
	indexed_errors 	= world.gather(errors_ind, root=0)
	indexed_errors  = [item for item in indexed_errors if item is not None]
	parameter_list 	= world.gather(parameters, root=0)
	parameter_list  = [item for item in parameter_list if item is not None]

	# Master does selection of elite parameters and drawing new means 

	if world.rank ==0:
		parameter_list_dict = dict([(param['param_id'], param)\
							 for param in parameter_list[1:]])
		indexed_errors_arr = np.array(indexed_errors[1:])
		indexed_errors_arr = indexed_errors_arr[np.argsort(\
									-indexed_errors_arr[:,1])]
		
		number_N = len(indexed_errors_arr) - np.sum(np.isnan(indexed_errors_arr[:,1]))
		
		elite_errors_indexed = indexed_errors_arr[0: N_elite]

		weights = np.exp((elite_errors_indexed[:,1] - np.min(elite_errors_indexed[:,1]))\
										/ (np.max(elite_errors_indexed[:,1]) -np.min(elite_errors_indexed[:,1])))
		gamma_XEM = np.append(gamma_XEM,\
									 elite_errors_indexed[-1, 1])
		S_star = np.append(S_star,\
									 elite_errors_indexed[0, 1])

		error_gamma = gamma_XEM[d +t-1] \
									- gamma_XEM[d +t -2]
		error_S = S_star[int(d +t-1)]\
									- S_star[int(d +t -2)]

		means, cov = gen_param_moments(parameter_list_dict,\
								param_random_bounds,\
								elite_errors_indexed, weights)

	gc.collect()
	return Number_N, [means, cov], gamma_XEM, S_star, error_gamma, error_S

def gen_param_moments(parameter_list_dict, param_random_bounds,\
						 selected, weights):

	""" Estiamate params of a sampling distribution

	Parameters
	----------
	parameter_list_dict: Dict
						  Dictionary with all paramameters
						  with ID keys
	selected            : 2D-array
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
				parameter_list_dict[int(selected[i,0])][key])
		
		sample_params.append(rand_params_i)

	sample_params = np.array(sample_params)
	means   = np.average(sample_params, weights = weights, axis=0)
	cov     = np.cov(sample_params, aweights =weights, rowvar=0)

	return means, cov


if __name__ == "__main__":

	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)

	world = MPI4py.COMM_WORLD
	world_size = world.Get_size()
	world_rank = world.Get_rank()
	block_size_layer_1 = 4
	block_size_layer_2 = 2
	block_size_layer_ts = 8

	param_random_bounds = {}
	settings_folder = 'settings/'
	model_name = 'test'


	# Create communicators 
	layer_1_comm, layer_2_comm, layer_ts_comm = gen_communicators(world,
													block_size_layer_1,
													block_size_layer_2,
													block_size_layer_ts)
	# Load model settings 
	with open("{}settings.yml", "r") as stream:
		eggbasket_config = yaml.safe_load(stream)

	with open('{}random_param_bounds.csv'\
		.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],\
				row['UB']])

	# Load and prepare data moments 
	moments_data = pd.read_csv('{}moments_data.csv'\
					.format(settings_folder))
	moments_data = moments_data.drop('Unnamed: 0', axis=1)   

	# Estimation parameters  
	tol = 1E-8
	TSN = 250
	N_elite = 45
	d = 3
	start = time.time()
	U = pickle.load(open("/scratch/pv33/{}/seed_U.smms"\
			.format(model_name),"rb")) 
	
	# Load previous iteration's parameters settings and means
	gamma_XEM_old, S_star_old,t_old, sampmom = load_tm1_iter(model_name)


	pickle.dump([means, cov],open("/scratch/pv33/{}/latest_means_iter.smms".format(model_name),"wb"))
	pickle.dump(gamma_XEM,open("/scratch/pv33/{}/gamma_XEM.smms".format(model_name),"wb"))
	pickle.dump(S_star,open("/scratch/pv33/{}/S_star.smms".format(model_name),"wb"))
	pickle.dump(t,open("/scratch/pv33/{}/t.smms".format(model_name),"wb"))
		

	print("...iteration {} on {} cores, elite_gamma error are {} and elite S error are {}"\
	.format(t, number_N, error_gamma, error_S))

	convg = int(np.abs(max(S_star[-d:]) - min(S_star[-d:]))< tol)

	print("...stop_error is {}, convergence is {}".format(np.abs(max(S_star[-d:]) - min(S_star[-d:])), convg))

	means, cov = gen_param_moments(parameter_list_dict,\
						param_random_bounds,\
						elite_errors_indexed, weights)

	convg_cov = int(np.abs(np.max(cov))< tol )
	print("...cov error is {}, convergence is {}".format((np.abs(np.max(cov))), convg_cov))

	pickle.dump([means, cov],\
				open("/scratch/pv33/{}/latest_means_iter.smms".format(model_name),"wb"))
	pickle.dump(gamma_XEM,\
				open("/scratch/pv33/{}/gamma_XEM.smms".format(model_name),"wb"))
	pickle.dump(S_star,\
				open("/scratch/pv33/{}/S_star.smms".format(model_name),"wb"))

	pickle.dump(t,\
				open("/scratch/pv33/{}/t.smms".format(model_name),"wb"))

	print("...generated and saved sampling moments")
	print("...time elapsed: {} minutes".format((time.time()-start)/60))


	convg = int(np.abs(max(S_star[-d:]) - min(S_star[-d:]))< tol)

	convg_cov = int(np.abs(np.max(cov))< tol )



