import numpy as np

def rand_p_generator(
                    param_deterministic, # dictionary containing deterministic parameters
                    param_random_bounds, # dictionary containing random parameters 
                    param_random_means=0,  # array of means of sampling distribution
                    param_random_cov=0,  # array of means of sampling distribution
                    deterministic = 1,   # flag to make all parmaeters determinstic 
                    initial =1, # if initial then draws from uniform disitrubution within bounds 
                    ):

    """Function generates list of parameters with 
    random parameters generated for params in the dictionary "pram_bounds"
    and deterministic parmameters for params in the dictionary "parameters". 

    Note: parameters in param_bounds will override any deterministic parameters in parameters
    If you want to make a parameter deterministic, remove it from the param_bounds list 
    """

    parameters = {}

    # first pull out all the parameters that are deterministic

    for key in param_deterministic:
        parameters[key] = param_deterministic[key]

    random_param_bounds_ar = \
        np.array([(bdns) for key, bdns in param_random_bounds.items()] ) 

    # generate random sample

    if deterministic == 0 and initial == 0:

        error_flag = np.random.uniform(0,1)
        if error_flag<.3:
            draws = np.random.multivariate_normal(param_random_means, param_random_cov*10)

            draws = np.clip(draws, random_param_bounds_ar[:,0],random_param_bounds_ar[:,1])

            for i,key in zip(np.arange(len(draws)),param_random_bounds.keys()):
                parameters[key]  = draws[i]
        elif error_flag<.9:
            draws = np.random.multivariate_normal(param_random_means, param_random_cov)

            draws = np.clip(draws, random_param_bounds_ar[:,0],random_param_bounds_ar[:,1])

            for i,key in zip(np.arange(len(draws)),param_random_bounds.keys()):
                parameters[key]  = draws[i]
        else:
            for key in param_random_bounds:
                parameters[key]  = np.random.uniform(param_random_bounds[key][0], param_random_bounds[key][1])
            

    if deterministic == 0 and initial == 1:
        for key in param_random_bounds:
            parameters[key]  = np.random.uniform(param_random_bounds[key][0], param_random_bounds[key][1])

    parameters['ID'] = np.random.randint(0,999999999999)

    return parameters