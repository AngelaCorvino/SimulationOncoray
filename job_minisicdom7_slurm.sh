
#!/bin/bash

#SBATCH --job-name=miniscidom7
#SBATCH --time=96:00:00
#SBATCH --ntasks=1

#export PATH=$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR


topas ./mini.txt
