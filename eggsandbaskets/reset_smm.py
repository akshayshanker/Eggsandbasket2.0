import dill as pickle
from tabulate import tabulate   
import numpy as np
import csv
import os


mod_name = 'final_male_v1'
TSN = 500
d = 3
path = "/scratch/pv33/ls_model_temp/{}".format(mod_name)

os.makedirs(path, exist_ok=True) 

t = 0
pickle.dump(t, open("/scratch/pv33/ls_model_temp/{}/t.smms".format(mod_name),"wb")) 
S_star,gamma_XEM	= np.full(d,0), np.full(d,0)
sampmom = [0,0]

U = np.random.rand(6,100,TSN,100)   

pickle.dump(U,open("/scratch/pv33/ls_model_temp/{}/seed_U.smms"\
			.format(mod_name),"wb"))

pickle.dump(gamma_XEM,open("/scratch/pv33/ls_model_temp/{}/gamma_XEM.smms"\
					.format(mod_name),"wb"))

pickle.dump(S_star,open("/scratch/pv33/ls_model_temp/{}/S_star.smms"\
					.format(mod_name),"wb"))

pickle.dump(sampmom,open("/scratch/pv33/ls_model_temp/{}/latest_sampmom.smms"\
					.format(mod_name),"wb"))

