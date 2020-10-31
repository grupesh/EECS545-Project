#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --account=yuni99
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --mem=64g
#SBATCH --mail-user=grupesh@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --output=./%x-%j
if [[ $SLURM_JOB_NODELIST ]] ; then
echo "Running on"
scontrol show hostnames $SLURM_JOB_NODELIST
fi

ml python3.7-anaconda/2020.02
export PYTHONPATH=$PWD:$PYTHONPATH
rm -r '../saved_data'
python3.7 gen_test_annotations.py
python3.7 gen_validation_set.py
python3.7 train.py --batch 30 --batch_vald 4 --lr 1e-3 --epochs 8000
