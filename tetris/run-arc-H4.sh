#!/bin/bash

#SBATCH -N1
#SBATCH --ntasks-per-node=10
#SBATCH -t 0-144:00:00
#SBATCH -p normal_q
#SBATCH -A qc_group

#load module
module reset
module load Anaconda3/2020.11
#module load Anaconda/5.2.0
module list
conda activate adapt

#run

for i in {1..5}
do
    nohup python trial_tetris.py $i > output_file/H4-tetris-r${i}.out &
done

for i in {1..5}
do
    nohup python trial_regular.py $i > output_file/H4-reg-r${i}.out &
done

wait;
exit;
