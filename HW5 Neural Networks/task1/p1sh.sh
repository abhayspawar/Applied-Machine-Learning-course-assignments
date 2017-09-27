#!/bin/sh
#SBATCH --account=edu            # The account name for the job.
#SBATCH --gres=gpu:1             # Request 1 gpu module
#SBATCH -c 1
#SBATCH --time=2:00:00           # Run time requested in hours:minutes:seconds

module load cuda80/toolkit cuda80/blas cudnn/5.1
module load anaconda/2-4.2.0

python /rigel/edu/coms4995/users/vb2428/task1/p1.py

