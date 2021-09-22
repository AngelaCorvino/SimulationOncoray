#!/bin/bash

#SBATCH --job-name=job2
#SBATCH --time=96:00:00
#SBATCH --ntasks=1

#export PATH=$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR


topas ./job2.txt
