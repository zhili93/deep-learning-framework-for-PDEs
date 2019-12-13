#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=48:40:00

# Use 16 MPI tasks:
#SBATCH -n 4
##SBATCH --exclusive
##SBATCH --exclude=node[856-1064]
##SBATCH -N 12
##SBATCH --ntasks-per-node=16
##SBATCH --constraint=sandy
##SBATCH --mincpus=16
#SBATCH --mem=30G

#SBATCH --mem-per-cpu=3G
# Specify a job name:
#SBATCH -J run

# Specify an output file
#SBATCH -o MyMPIJob-%j.out
#SBATCH -e MyMPIJob-%j.error
ncpus=4

module load python/3.7.4

source /gpfs_home/zli6/virtual_env/env/bin/activate

python NavierStokes_tf2_learning_rate.py

