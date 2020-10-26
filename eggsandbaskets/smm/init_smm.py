import numpy as np
import dill as pickle 


d = 3
TSN = 250


model_name = 'test'

S_star,gamma_XEM	= np.full(d,0), np.full(d,0)
t, stop     		= 0,0 
error_gamma 		= 0
error_S 			= 0

U = np.random.rand(6,100,TSN,100)    


pickle.dump(gamma_XEM,open("/scratch/pv33/{}/gamma_XEM.smms".fromat(model_name),"wb"))
pickle.dump(S_star,open("/scratch/pv33/{}/S_star.smms".fromat(model_name),"wb"))
pickle.dump(t, open("/scratch/pv33/{}/t.smms".fromat(model_name),"wb"))
pickle.dump(U, open("/scratch/pv33/{}/seed_U.smms".fromat(model_name),"wb"))