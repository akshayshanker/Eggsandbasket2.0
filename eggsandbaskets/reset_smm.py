import dill as pickle
from tabulate import tabulate   
import numpy as np
import csv
import os
import sys


mod_name  = sys.argv[1]
TSN = 500
d = 3
path = "/scratch/pv33/ls_model_temp2/{}".format(mod_name)

os.makedirs(path, exist_ok=True) 

t = 0
pickle.dump(t, open(path + "/t.smms","wb")) 
#sampmom = pickle.load(open("/scratch/pv33/ls_model_temp2/{}/latest_sampmom.smms".format('final_female_v1'),"rb")) 
sampmom = [0,0]
pickle.dump(sampmom,open(path + "/latest_sampmom.smms","wb"))


S_star,gamma_XEM	= np.full(d,0), np.full(d,0)


U = np.random.rand(6,100,TSN,100)   

pickle.dump(U,open(path + "/seed_U.smms","wb"))

pickle.dump(gamma_XEM,open(path + "/gamma_XEM.smms","wb"))

pickle.dump(S_star,open(path + "/S_star.smms","wb"))

pickle.dump(sampmom,open(path + "/latest_sampmom.smms","wb"))

