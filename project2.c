#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

//update after each modification
#define VERSION 1.03

//global variables
int g_writeToFile = 0;
int g_threadsPerProc = 64;
int g_numNodes = 4;
int g_numRows = 100000;

//dimension of the matrix
#define ROW_COLUMN_NUM g_numRows
#define ROW_NUM ROW_COLUMN_NUM
#define COLUMN_NUM ROW_COLUMN_NUM

#define RUNTIME

//the number of threads for generating matrix(no need to modify)
#define GENERATE_THREAD_NUM 32

//block_size for static schedule
#define BLOCK_SIZE (ROW_NUM/32)

typedef struct CompressedMatrix {
  int** B;
  int** C;
  int* rowLengths;
  int maxRowLength;
  int numRow;
}CompressedMatrix;

int randomNumber(unsigned int* seed) {
#ifdef _WIN32
  return rand_s();
#else 
  return rand_r(seed);
#endif
}

int** alloc_matrix(int row, int column){
    omp_set_num_threads(GENERATE_THREAD_NUM);
    int** matrix = malloc(sizeof(int*)*row);
    int i = 0;
#pragma omp parallel for
    for(i = 0 ; i < row; ++i){
        matrix[i] = calloc(column, sizeof(int));
    }
    return matrix;
}

void free_matrix(int** matrix, int row){
    omp_set_num_threads(GENERATE_THREAD_NUM);
    int i = 0;
#pragma omp parallel for private(i)
    for(i = 0 ;i < row;++i){
      free(matrix[i]);
    }
    free(matrix);
}

void zero_matrix(int** matrix, int row, int column){
    omp_set_num_threads(GENERATE_THREAD_NUM);
    int i = 0;
#pragma omp parallel for private(i)
    for(i = 0; i<row; ++i){
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
        int i = 0;
#pragma omp for private(i)
        for (i = 0; i < size; ++i) {
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
  int i = 0;
#pragma omp parallel for private(i)
  for (i = 0; i < numRow; ++i) {
    for (int j = 0; j < numColumn; ++j) {
      B[i][j] = 0;
      C[i][j] = 0;
    }
  }

#pragma omp parallel for private(i)
  for (i = 0; i < numRow; ++i) {
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
    B[i] = realloc(B[i], sizeof(int) * col);
    C[i] = realloc(C[i], sizeof(int) * col);
  }

  ret.B = B;
  ret.C = C;
  
  return ret;
}

void fill_compressed_matrix(CompressedMatrix* mat, double probability, int sd) {
  omp_set_num_threads(g_threadsPerProc);
#pragma omp parallel
  {
    //printf("thread_id:%d\n", omp_get_thread_num());
    int seed = sd * g_threadsPerProc + omp_get_thread_num();
    int i = 0;
#pragma omp for private(i)
    for (i = 0; i < ROW_NUM; ++i) {
      int nonZeroNum = 0;
      int* data = calloc(COLUMN_NUM, sizeof(int));
      int* index = calloc(COLUMN_NUM, sizeof(int));
      for (int j = 0; j < COLUMN_NUM; ++j) {
        if (((double)randomNumber(&seed) / RAND_MAX) < probability) {
          data[nonZeroNum] = (randomNumber(&seed) % 10) + 1;
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

        int i = 0;
#pragma omp for private(i) schedule(static, BLOCK_SIZE) 
        for (i = iter_start; i < iter_end; ++i) {
            for (int k = 0; k < X.rowLengths[i]; ++k) {
                int index = X.C[i][k];
                for (int l = 0; l < Y.rowLengths[index]; ++l) {
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

  return 0;
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
      MPI_Send(X.B[i], X.rowLengths[i], MPI_INT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(X.C[i], X.rowLengths[i], MPI_INT, dest, 0, MPI_COMM_WORLD);
    }
  } else {
    for (int i = rank * rowEachNode; i < (rank+1)* rowEachNode; ++i) {
      X.B[i] = calloc(X.rowLengths[i], sizeof(int));
      MPI_Recv(X.B[i], X.rowLengths[i], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      X.C[i] = calloc(X.rowLengths[i], sizeof(int));
      MPI_Recv(X.C[i], X.rowLengths[i], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
  }

  //broadcast Y to others
  for (int i = 0; i < ROW_NUM; ++i) {
    MPI_Bcast(Y.B[i], Y.rowLengths[i], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y.C[i], Y.rowLengths[i], MPI_INT, 0, MPI_COMM_WORLD);
  }
    
  //openmpi matrix multiplication
  printf("start_omp_matrix_multiplication\n");
  int** output = omp_matrix_multiply(X, Y, g_threadsPerProc, rank, numNode);
  printf("transform_sparse_to_compressed_matrix\n");
  CompressedMatrix result = transform_sparse_to_compressed_matrix(rowEachNode, COLUMN_NUM, output);
  printf("free output\n");
  free_matrix(output, rowEachNode);


  //recv result
  printf("recv result\n");
  CompressedMatrix finalResult;
  if (rank == 0) {
    initCompressedMatrix(&finalResult);
  }

  //send result length to master
  printf("send result length to master\n");
  if (rank != 0) {
    MPI_Send(result.rowLengths, rowEachNode, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    for (int i = 1; i < numNode; ++i) {
      MPI_Recv(finalResult.rowLengths + rowEachNode * i, rowEachNode, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
    }
    for (int j = 0; j < rowEachNode; ++j) {
      finalResult.rowLengths[j] = result.rowLengths[j];
    }
  }

  //send result to master
  printf("send result to master\n");
  if (rank != 0) {
    for (int i = 0; i < rowEachNode; ++i) {
      MPI_Send(result.B[i], result.rowLengths[i], MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(result.C[i], result.rowLengths[i], MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }
  //master receive results
  if (rank == 0) {
    for (int j = 0; j < rowEachNode; ++j) {
      for (int i = 1; i < numNode; ++i) {
        int index = i * rowEachNode + j;
        finalResult.B[index] = calloc(finalResult.rowLengths[index], sizeof(int));
        MPI_Recv(finalResult.B[index], finalResult.rowLengths[index], MPI_INT, i, 0, MPI_COMM_WORLD, &status);
        finalResult.C[index] = calloc(finalResult.rowLengths[index], sizeof(int));
        MPI_Recv(finalResult.C[index], finalResult.rowLengths[index], MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      }
    }
    for (int j = 0; j < rowEachNode; ++j) {
      finalResult.B[j] = result.B[j];
      finalResult.C[j] = result.C[j];
    }
  }

  //write  to file (do we need to support parallel writing? TODO)
  if (g_writeToFile != 0 && rank == 0) {
    printf("write to file\n");
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
    double probability = 0.01;
    if (argc > 1) {
      g_writeToFile = atoi(argv[1]);
    }
    if (argc > 2) {
      g_threadsPerProc = atoi(argv[2]);
    }
    if (argc > 3) {
      g_numRows = atoi(argv[3]);
    }
    if (argc > 4) {
      probability = atof(argv[4]);
    }
    if (argc > 5) {
      g_numNodes = atoi(argv[5]);
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
      printf("matrix_rows:%d\n", ROW_NUM);
      printf("mpi_numNodes:%d\n", size);
      printf("thread_per_proc:%d\n", g_threadsPerProc);
    }
    
    sample(0, probability, rank, size);

    MPI_Finalize();

    return 0;
}


