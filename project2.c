#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

//dimension of the matrix
#define ROW_COLUMN_NUM ((int64_t)100000)
#define ROW_NUM ROW_COLUMN_NUM
#define COLUMN_NUM ROW_COLUMN_NUM

#define RUNTIME

//the number of threads for generating matrix(no need to modify)
#define GENERATE_THREAD_NUM 32

//block_size for static schedule
#define BLOCK_SIZE (ROW_NUM/32)

//global variables
int g_writeToFile = 0;
int g_threadsPerProc = 64;
int g_numNodes = 4;

typedef struct CompressedMatrix {
  int** B;
  int** C;
  int* rowLengths;
  int maxRowLength;
  int numRow;
}CompressedMatrix;

int** alloc_matrix(int row, int column){
    omp_set_num_threads(GENERATE_THREAD_NUM);
    int** matrix = malloc(sizeof(int*)*row);
#pragma omp parallel for
    for(int i = 0 ; i < row; ++i){
        matrix[i] = calloc(column, sizeof(int));
    }
    return matrix;
}

void free_matrix(int** matrix, int row){
    omp_set_num_threads(GENERATE_THREAD_NUM);
#pragma omp parallel for
    for(int i = 0 ;i < row;++i){
      free(matrix[i]);
    }
    free(matrix);
}

void zero_matrix(int** matrix, int row, int column){
    omp_set_num_threads(GENERATE_THREAD_NUM);
#pragma omp parallel for
    for(int i= 0; i<row; ++i){
        for(int j =0; j< column; ++j){
            matrix[i][j] = 0;
        }
    }
}

void free_compressed_matrix(CompressedMatrix* m) {
    if (m != NULL) {
        for (int i = 0; i < m->numRow; ++i) {
          free(m->B[i]);
          free(m->C[i]);
        }
        free(m->rowLengths);
        m->maxRowLength = 0;
        m->numRow = 0;
        free(m->B);
        free(m->C);
    }
}

int** matrix_multiply(int **A, int **B, int size) {
    int** C = alloc_matrix(size, size);
    omp_set_num_threads(GENERATE_THREAD_NUM);
    printf("matrix_multiply_start\n");
#pragma omp parallel
    {
        printf("thread_id:%d\n", omp_get_thread_num());
#pragma omp for
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < size; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    printf("matrix_multiply_end\n");
    return C;
}

CompressedMatrix transform_sparse_to_compressed_matrix(int numRow, int numColumn, int** A) {
  CompressedMatrix ret;
  ret.numRow = numRow;
  int** B = alloc_matrix(numRow, numColumn);
  int** C = alloc_matrix(numRow, numColumn);
  ret.rowLengths = calloc(numRow, sizeof(int));
  ret.maxRowLength = 0;

  omp_set_num_threads(GENERATE_THREAD_NUM);
#pragma omp parallel for
  for (int i = 0; i < numRow; ++i) {
    for (int j = 0; j < numColumn; ++j) {
      B[i][j] = 0;
      C[i][j] = 0;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < numRow; ++i) {
    int col = 0;
    B[i] = calloc(numColumn, sizeof(int));
    C[i] = calloc(numColumn, sizeof(int));
    for (int j = 0; j < numColumn; ++j) {
      if (A[i][j] != 0) {
        B[i][col] = A[i][j];
        C[i][col] = j;
        col++;
      }
    }
    if (col > ret.maxRowLength) {
      ret.maxRowLength = col;
    }
    ret.rowLengths[i] = col;
    realloc(B[i], sizeof(int) * col);
    realloc(C[i], sizeof(int) * col);
  }

  ret.B = B;
  ret.C = C;
  
  return ret;
}

void fill_compressed_matrix(CompressedMatrix* mat, double probability, int sd) {
#pragma omp parallel
  {
    //printf("thread_id:%d\n", omp_get_thread_num());
    int seed = sd + omp_get_thread_num();
#pragma omp for
    for (int i = 0; i < ROW_NUM; ++i) {
      int nonZeroNum = 0;
      int* data = calloc(COLUMN_NUM, sizeof(int));
      int* index = calloc(COLUMN_NUM, sizeof(int));
      for (int j = 0; j < COLUMN_NUM; ++j) {
        if (((double)rand_r(&seed) / RAND_MAX) < probability) {
          data[nonZeroNum] = (rand_r(&seed) % 10) + 1;
          index[nonZeroNum] = j;
          nonZeroNum += 1;
        }
      }
      mat->maxRowLength = nonZeroNum > mat->maxRowLength ? nonZeroNum : mat->maxRowLength;
      mat->rowLengths[i] = nonZeroNum;
      mat->B[i] = realloc(data, sizeof(int)* nonZeroNum);
      mat->C[i] = realloc(index, sizeof(int) * nonZeroNum);
    }
  }

  return;
}

int** omp_matrix_multiply(CompressedMatrix X, CompressedMatrix Y, int numThread, int rank, int numNode){
    int numRows = ROW_NUM / numNode;
    int** matrix = alloc_matrix(numRows, COLUMN_NUM);
  
    printf("start_parallel\n");
    omp_set_num_threads(numThread);
    double start = omp_get_wtime();

    int iter_start = rank * numRows;
    int iter_end = (rank + 1) * numRows;

#pragma omp parallel 
    {
#pragma omp master
        {
            printf("Number of threads in parallel region: %d\n", omp_get_num_threads());   
        }
        //printf("Thread_id:%d\n", omp_get_thread_num());

#pragma omp for schedule(static, BLOCK_SIZE)
        for (int i = iter_start; i < iter_end; ++i) {
            for (int k = 0; k < X.rowLengths[i]; ++k) {
                int index = X.C[i][k];
                for (int l = 0; l < Y.rowLengths[i]; ++l) {
                    matrix[i- iter_start][Y.C[index][l]] += X.B[i][k] * Y.B[index][l];
                }
            }
        }

    }
    double end = omp_get_wtime();
    double timeSpent = (double)(end - start);
    printf("time spent = %10.6f\n", timeSpent);

    return matrix;
}

int mpi_parallel_write(int rank, int rowEachNode, CompressedMatrix m, const char* filenameB, const char* filenameC) {
  int startRow = rank * rowEachNode;
  int endRow = (rank + 1) * rowEachNode;
  //calculate lengths;
  char* buffer = calloc(COLUMN_NUM * 4, 1);
  int pos = 0;
}

int mpi_independent_write(CompressedMatrix m, const char* filenameB, const char* filenameC) {
  MPI_File fileB;
  int err = MPI_File_open(MPI_COMM_WORLD, filenameB, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileB);
  MPI_File fileC;
  int err1 = MPI_File_open(MPI_COMM_WORLD, filenameC, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileC);
  if (err == MPI_SUCCESS && err1 == MPI_SUCCESS) {
    char* buffer = calloc(COLUMN_NUM * 4, 1);
    int pos = 0;

    for (int k = 0; k < m.numRow; k++) {
      pos = 0;
      int size = m.rowLengths[k];
      for (int i = 0; i < size; ++i) {
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "%d", m.B[k][i]);
        if (i < size - 1) {
          pos += snprintf(buffer + pos, sizeof(buffer) - pos, ",");
        }
      }
      snprintf(buffer + pos, sizeof(buffer) - pos, "\n");
      MPI_File_write(fileB, buffer, pos + 1, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    for (int k = 0; k < m.numRow; k++) {
      pos = 0;
      int size = m.rowLengths[k];
      for (int i = 0; i < size; ++i) {
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "%d", m.C[k][i]);
        if (i < size - 1) {
          pos += snprintf(buffer + pos, sizeof(buffer) - pos, ",");
        }
      }
      snprintf(buffer + pos, sizeof(buffer) - pos, "\n");
      MPI_File_write(fileC, buffer, pos + 1, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    free(buffer);
  }
  else {
    //TODO
  }
  MPI_File_close(&fileB);
  MPI_File_close(&fileC);
}

void initCompressedMatrix(CompressedMatrix* mat) {
  mat->B = calloc(ROW_NUM, sizeof(int*));
  mat->C = calloc(ROW_NUM, sizeof(int*));
  mat->numRow = ROW_NUM;
  mat->rowLengths = calloc(ROW_NUM, sizeof(int));
  mat->maxRowLength = 0;
}

void sample(int id, double probability, int rank, int numNode) {
  printf("sample%d:\n", id);
  MPI_Status status;
  int rowEachNode = ROW_NUM / numNode;
  CompressedMatrix X;
  initCompressedMatrix(&X);
  CompressedMatrix Y;
  initCompressedMatrix(&Y);
  if (rank == 0) {
    fill_compressed_matrix(&X, probability, id);
    fill_compressed_matrix(&Y, probability, id);
  }

  //send X and Y rowLengths to others
  MPI_Bcast(X.rowLengths, ROW_NUM, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(Y.rowLengths, ROW_NUM, MPI_INT, 0, MPI_COMM_WORLD);

  //send X
  if (rank == 0) {
    //split X to rows and send to different nodes 
    for (int i = rowEachNode; i < ROW_NUM; ++i) {
      int dest = i / rowEachNode;
      int tag = i;
      MPI_Send(X.B[i], X.rowLengths[i], MPI_INT, dest, tag, MPI_COMM_WORLD);
      MPI_Send(X.C[i], X.rowLengths[i], MPI_INT, dest, -tag - 1, MPI_COMM_WORLD);
    }
  } else {
    for (int i = rank * rowEachNode; i < (rank+1)* rowEachNode; ++i) {
      X.B[i] = calloc(X.rowLengths[i], sizeof(int));
      MPI_Recv(X.B[i], X.rowLengths[i], MPI_INT, 0, i, MPI_COMM_WORLD, &status);
      X.C[i] = calloc(X.rowLengths[i], sizeof(int));
      MPI_Recv(X.C[i], X.rowLengths[i], MPI_INT, 0, -i - 1, MPI_COMM_WORLD, &status);
    }
  }

  //broadcast Y to others
  for (int i = 0; i < ROW_NUM; ++i) {
    MPI_Bcast(Y.B[i], Y.rowLengths[i], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y.C[i], Y.rowLengths[i], MPI_INT, 0, MPI_COMM_WORLD);
  }
    
  //openmpi matrix multiplication
  int** output = omp_matrix_multiply(X, Y, g_threadsPerProc, rank, numNode);
  CompressedMatrix result = transform_sparse_to_compressed_matrix(rowEachNode, COLUMN_NUM, output);
  free_matrix(output, rowEachNode);


  //recv result
  CompressedMatrix finalResult;
  if (rank == 0) {
    initCompressedMatrix(&finalResult);
  }

  //send result length to master
  if (rank != 0) {
    MPI_Send(result.rowLengths, rowEachNode, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    for (int i = 1; i < numNode; ++i) {
      MPI_Recv(finalResult.rowLengths + rowEachNode * rank, rowEachNode, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
    }
    for (int j = 0; j < rowEachNode; ++j) {
      finalResult.rowLengths[j] = result.rowLengths[j];
    }
  }

  //send result to master
  if (rank != 0) {
    for (int i = 0; i < rowEachNode; ++i) {
      MPI_Send(result.B[i], result.rowLengths[i], MPI_INT, 0, rank, MPI_COMM_WORLD);
    }
    for (int i = 0; i < rowEachNode; ++i) {
      MPI_Send(result.C[i], result.rowLengths[i], MPI_INT, 0, -rank, MPI_COMM_WORLD);
    }
  }

  //master receive results
  if (rank == 0) {
    for (int i = 1; i < numNode; ++i) {
      for (int j = 0; j < rowEachNode; ++j) {
        finalResult.B[i * rowEachNode + j] = calloc(finalResult.rowLengths[i * rowEachNode + j], sizeof(int));
        MPI_Recv(finalResult.B[i * rowEachNode + j], finalResult.rowLengths[i * rowEachNode + j], MPI_INT, i, i, MPI_COMM_WORLD, &status);
      }
      for (int j = 0; j < rowEachNode; ++j) {
        finalResult.C[i * rowEachNode + j] = calloc(finalResult.rowLengths[i * rowEachNode + j], sizeof(int));
        MPI_Recv(finalResult.C[i * rowEachNode + j], finalResult.rowLengths[i * rowEachNode + j], MPI_INT, i, -i, MPI_COMM_WORLD, &status);
      }
    }
    for (int j = 0; j < rowEachNode; ++j) {
      finalResult.B[j] = result.B[j];
      finalResult.C[j] = result.C[j];
    }
  }

  //write  to file (do we need to support parallel writing? TODO)
  if (g_writeToFile != 0 && rank == 0) {
    mpi_independent_write(X, "XB.txt", "XC.txt");
    mpi_independent_write(Y, "YB.txt", "YC.txt");
    mpi_independent_write(finalResult, "XYB.txt", "XYC.txt");
  }
  else {

  }

  //free memory
  printf("start_free_memory\n");
  if (rank != 0) {
    free_compressed_matrix(&result);
  }
  if (rank == 0) {
    free(result.B);
    free(result.C);
    free(result.rowLengths);
  }
  free_compressed_matrix(&X);
  free_compressed_matrix(&Y);

  if (rank == 0) {
    free_compressed_matrix(&finalResult);
  }
}


int main(int argc, char* argv[]){
    //TODO: test

    /*
    argv[1]:write to file or not, if not exsit, not write to file
    argv[2]:number of threads in each node to be used, default 64
    argv[3]:number of nodes(TODO)
    */
    if (argc > 1) {
      g_writeToFile = atoi(argv[1]);
    }
    if (argc > 2) {
      g_threadsPerProc = atoi(argv[2]);
    }
    if (argc > 3) {
      g_numNodes = atoi(argv[3]);
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sample(1, 0.01, rank, size);
    sample(2, 0.02, rank, size);
    sample(3, 0.05, rank, size);

    MPI_Finalize();

    return 0;
}


