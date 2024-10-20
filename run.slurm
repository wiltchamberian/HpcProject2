#!/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=work
#SBATCH --account=courses0101
#SBATCH --mem=200G
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=128

cc -fopenmp project2.c -o project2
srun ./project2 0 128 100000 0.05
