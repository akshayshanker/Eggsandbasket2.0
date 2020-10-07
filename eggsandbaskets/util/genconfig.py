
import numpy as np
import pandas as pd
from numpy import genfromtxt
import csv
import numba 
from quantecon import tauchen
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
import time
import dill as pickle 
from randparam   import rand_p_generator
import copy
import sys

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd

import yaml 

# housing model modules 
from pointsgen import generate_points
from mathfuncs import *
from worker_operators import housingmodel_operator_factory
from housing_functions import housingmodel_function_factory
from retiree_operators import housing_model_retiree_func_factory
from profiles_moments import genprofiles_operator, gen_moments,\
                             sortmoments


normalisation = np.array([1, 1E-5])
param_deterministic = {}
parameter_description = {}
param_random_bounds = {}
N = 480

settings_folder = '/home/141/as3442/Retirementeggs/settings'

with open('{}/parameters_EGM_base.csv'.format(settings_folder),\
    newline='') as pscfile:
    reader = csv.DictReader(pscfile)
    for row in reader:
        param_deterministic[row['parameter']] = float(row['value'])
        parameter_description[row['parameter']] =row['description']

with open('{}/random_param_bounds.csv'\
    .format(settings_folder), newline='') as pscfile:
    reader_ran = csv.DictReader(pscfile)
    for row in reader_ran:        #print(row['UB'])
        param_random_bounds[row['parameter']] = [row['LB'],\
            row['UB']]

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

# Load latest estimates

estimates = pickle.load(open("/scratch/pv33/latest_means_iter.smms","rb")) 


for i,key in zip(np.arange(len(estimates[0])),param_random_bounds.keys()):
        parameters[key]  = float(estimates[0][i])

eggbasket_config = {}

eggbasket_config['baseline_lite'] = {}

eggbasket_config['baseline_lite']['parameters'] = parameters
eggbasket_config['baseline_lite']['lambdas'] = lambdas.tolist()
eggbasket_config['baseline_lite']['survival'] = survival.tolist()
eggbasket_config['baseline_lite']['vol_cont_points'] = vol_cont_points.tolist()
eggbasket_config['baseline_lite']['risk_share_points'] = risk_share_points.tolist()

eggbasket_config['baseline_lite']['parameter_description'] = parameter_description

with open('eggbasket_config.yml', 'w') as outfile:
	yaml.dump(eggbasket_config, outfile) 


