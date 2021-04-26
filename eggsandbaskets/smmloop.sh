#!/bin/bash
cd $HOME
cd Eggsandbaskets
cd eggsandbaskets  
module load python3/3.7.4
module load openmpi/4.0.2

UCX_LOG_LEVEL=error

for var in {1..50}
do
	 mpiexec -n 1920  python3 -m mpi4py smm.py male final_male_v2
done 

