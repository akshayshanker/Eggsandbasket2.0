  
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

from profiles_moments import genprofiles_operator, gen_moments, sortmoments
from housing_functions import housingmodel_function_factory
from egg_basket import HousingModel, housingmodel_operator_factory
from retiree_operators import housing_model_retiree_func_factory

from pyina import mpi
world = mpi.world

from mpi4py import MPI as MPI4py
comm = MPI4py.COMM_WORLD

from pyina.mpi_pool import parallel_map

import gc


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

def gen_RMS(parameters,lambdas,\
			survival,\
			moments_data,\
			vol_cont_points,\
			risk_share_points, TSN,U):
	"""
	Generate root mean square error 
	between simulated moments for HousingModel 
	and data moments 

	"""
	# define functions 

	functions = {}

	functions['u'], functions['uc'], functions['uh'], functions['b'], \
	functions['b_prime'], functions['y'],functions['yvec'], functions['DB_benefit'], \
	functions['adj_p'], functions['adj_v'], functions['adj_pi'],\
	functions['uc_inv'],functions['uh_inv'],\
		= housingmodel_function_factory(parameters,\
										 lambdas,\
										  normalisation)

	# Create housing model 
	og              = HousingModel(functions, parameters, survival,\
										vol_cont_points,\
										risk_share_points)
	
	# solve model 
	gen_R_pol       = housing_model_retiree_func_factory(og)

	solve_LC_model  = housingmodel_operator_factory(og,gen_R_pol)

	policies       	= (solve_LC_model())

	# generate time series 
	generate_TSDF     = genprofiles_operator(og)

	del og
	gc.collect() 

	TS1, TS2        = generate_TSDF(U,TSN, *policies)

	# generate and sort moments

	moments_sim_array, moments_data_array \
	= gen_format_moments(TS1, TS2, moments_data)

	del TS1
	del TS2 
	gc.collect()

	deviation = (moments_sim_array\
								[~np.isnan(moments_data_array)]\
								  - moments_data_array\
								  [~np.isnan(moments_data_array)])

	norm  = np.sum(np.square(moments_data_array[~np.isnan(moments_data_array)]))



	N_err = len(deviation)

	return 1-np.sqrt((1/N_err)*np.sum(np.square(deviation))/norm)

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



	normalisation = np.array([1E-5, 100])
	param_deterministic = {}
	param_random_bounds = {}

	settings_folder = '/home/141/as3442/Retirementeggs/settings'


	# un-pack model settings 

	with open('{}/parameters_EGM_base.csv'.format(settings_folder),\
		newline='') as pscfile:
		reader = csv.DictReader(pscfile)
		for row in reader:
			param_deterministic[row['parameter']] = np.float64(row['value'])

	with open('{}/random_param_bounds.csv'\
		.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
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
	

	# load and prepare data moments 
	moments_data = pd.read_csv('{}/moments_data.csv'\
					.format(settings_folder))
	moments_data = moments_data.drop('Unnamed: 0', axis=1)   

	# run SMM estimation 

	tol   	= 1E-8
	TSN   	= 150
	N_elite = 45
	d     	= 3

	start = time.time()
	# pick previous parameters settings and means
	gamma_XEM 	= pickle.load(open("/scratch/pv33/gamma_XEM.smms","rb"))
	S_star 		= pickle.load(open("/scratch/pv33/S_star.smms","rb"))
	t 			= pickle.load(open("/scratch/pv33/t.smms","rb"))

	sampmom 	= pickle.load(open("/scratch/pv33/latest_means_iter.smms","rb"))

	# generate new parameter sample (each worker generates a random sample)

	if t ==0:
		initial =0
	else:
		initial = 0

	parameters             = rand_p_generator(param_deterministic,\
								param_random_bounds, deterministic = 0,\
								initial =initial,\
								param_random_means = sampmom[0],\
								param_random_cov = sampmom[1])

	t = t+1

	indexed_errors = None
	parameter_list = None

	# eval model on each worker 
	if world.rank ==0:
		print("Distributng iter {}".format(t))
		errors_ind = [0,0]
	else:
		def SMM_objective():
			"""SMM objective to be maximised 
			as function of params"""
			parameters_all = parameters
			U = pickle.load(open("/scratch/pv33/seed_U.smms","rb")) 
			#U = np.random.rand(6,100,TSN,100) 
			RMS =  gen_RMS(parameters_all,lambdas,\
						survival,\
						moments_data,\
						vol_cont_points,\
						risk_share_points,TSN,U)

			return [parameters_all['ID'], RMS]
		errors_ind = SMM_objective()
		del SMM_objective
		gc.collect()

	comm.Barrier()
	indexed_errors 	= comm.gather(errors_ind, root=0)
	parameter_list 	= comm.gather(parameters, root=0)

	# master does calculations
	
	if world.rank ==0:
		parameter_list_dict = dict([(param['ID'], param)\
							 for param in parameter_list[1:]])
		indexed_errors_arr      = np.array(indexed_errors[1:])
		indexed_errors_arr      = indexed_errors_arr[np.argsort(\
									-indexed_errors_arr[:,1])]
		number_N 				= len(indexed_errors_arr) - np.sum(np.isnan(indexed_errors_arr[:,1]))


		elite_errors_indexed    = indexed_errors_arr[0: N_elite]

		weights 				= np.exp((elite_errors_indexed[:,1] - np.min(elite_errors_indexed[:,1]))\
										/ (np.max(elite_errors_indexed[:,1]) -np.min(elite_errors_indexed[:,1])))


		gamma_XEM               = np.append(gamma_XEM,\
									 elite_errors_indexed[-1, 1])
		S_star                  = np.append(S_star,\
									 elite_errors_indexed[0, 1])

		error_gamma             = gamma_XEM[d +t-1] \
									- gamma_XEM[d +t -2]
		error_S                 = S_star[int(d +t-1)]\
									- S_star[int(d +t -2)]

		print("...iteration {} on {} cores, elite_gamma error are {} and elite S error are {}"\
			.format(t, number_N, error_gamma, error_S))

		convg = int(np.abs(max(S_star[-d:]) - min(S_star[-d:]))< tol)

		print("...stop_error is {}, convergence is {}".format(np.abs(max(S_star[-d:]) - min(S_star[-d:])), convg))

		means, cov          = gen_param_moments(parameter_list_dict,\
								param_random_bounds,\
								elite_errors_indexed, weights)

		convg_cov		= int(np.abs(np.max(cov))< tol )
		print("...cov error is {}, convergence is {}".format((np.abs(np.max(cov))), convg_cov))

		pickle.dump([means, cov],\
						open("/scratch/pv33/latest_means_iter.smms","wb"))
		pickle.dump(gamma_XEM,\
						open("/scratch/pv33/gamma_XEM.smms","wb"))
		pickle.dump(S_star,\
						open("/scratch/pv33/S_star.smms","wb"))

		pickle.dump(t,\
						open("/scratch/pv33/t.smms","wb"))
		
		print("...generated and saved sampling moments")
		print("...time elapsed: {} minutes".format((time.time()-start)/60))

