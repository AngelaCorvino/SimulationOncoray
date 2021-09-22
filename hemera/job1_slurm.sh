#!/bin/bash

#SBATCH --job-name=job1
#SBATCH --time=96:00:00
#SBATCH --ntasks=1

#export PATH=$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR


topas ./job1.txt
