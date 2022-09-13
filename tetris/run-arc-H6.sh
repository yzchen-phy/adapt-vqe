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
    nohup python trial_tetris6.py $i > output_file/QEB-tetris-H6-r${i}.out &
done

for i in {1..5}
do
    nohup python trial_regular6.py $i > output_file/QEB-reg-H6-r${i}.out &
done

wait;
exit;
