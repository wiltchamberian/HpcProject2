# HpcProject2
uwa HPC project2

# command explanation

## set OMP_SCHEDULE
export OMP_SCHEDULE="static,200"

## compile with mpi and openmp
cc -fopenmp project2.c -o project2

## parameters
1. parameter1: whether writing to file or not (1:yes, 0:no)
2. parameter2: number of threads per process
3. parameter3: matrix row and column number
4. parameter4: probability of non-zero elements
5. parameter5: a global seed for generating random numbers
srun ./project2 0 128 100000 0.05 0