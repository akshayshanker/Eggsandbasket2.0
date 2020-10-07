#!/bin/bash


cd $HOME
cd Retirementeggs
cd retirementreggs  
module load python3/3.7.4
module load openmpi/4.0.2

alias mpython='mpiexec -oversubscribe -np 1800 `which python3`'
 
python3 init_SMM.py

for var in {1..50}
do
	mpython SMM2.py
done 

