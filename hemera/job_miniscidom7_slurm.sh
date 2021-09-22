#!/bin/bash

#SBATCH --job-name=miniscidom6lessparticle
#SBATCH --time=96:00:00     
#SBATCH --ntasks=1

#export PATH=$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
  

topas ./mini6_50000000.txt
