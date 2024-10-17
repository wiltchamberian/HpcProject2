#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

//update after each modification
#define VERSION 2.1

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

void initCompressedMatrix(CompressedMatrix* mat, int rowNum) {
  mat->B = calloc(rowNum, sizeof(int*));
  mat->C = calloc(rowNum, sizeof(int*));
  mat->numRow = rowNum;
  mat->rowLengths = calloc(rowNum, sizeof(int));
  mat->maxRowLength = 0;
}


int randomNumber(unsigned int* seed) {
#ifdef _WIN32
  return 7;
#else 
  return rand_r(seed);
#endif
}

int** alloc_matrix(int row, int column){
    omp_set_num_threads(GENERATE_THREAD_NUM);
    int** matrix = malloc(sizeof(int*) * row);
    if (matrix == NULL) {
      return NULL;
    }
    int i = 0;
#pragma omp parallel for private(i)
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

CompressedMatrix omp_matrix_multiply(CompressedMatrix X, CompressedMatrix Y, int numThread, int rank, int numNode){
    int numRows = ROW_NUM / numNode;

    CompressedMatrix result;
    initCompressedMatrix(&result, numRows);
  
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
        int i = 0;
#pragma omp for private(i) schedule(runtime) 
        for (i = iter_start; i < iter_end; ++i) {
            result.B[i - iter_start] = calloc(COLUMN_NUM, sizeof(int));
            result.C[i - iter_start] = calloc(COLUMN_NUM, sizeof(int));
            for (int k = 0; k < X.rowLengths[i]; ++k) {
                int index = X.C[i][k];
                for (int l = 0; l < Y.rowLengths[index]; ++l) {
                    result.B[ i- iter_start][Y.C[index][l]] += X.B[i][k] * Y.B[index][l];
                }
            }
            int col = 0;
            for (int s = 0; s < numRows; ++s) {
                if (result.B[i - iter_start][s] != 0) {
                    result.C[i - iter_start][col] = s;
                    result.B[i - iter_start][col] = result.B[i - iter_start][s];
                    col += 1;
                }
            }
            if (col > 0) {
              result.B[i - iter_start] = realloc(result.B[i - iter_start], col * sizeof(int));
              result.C[i - iter_start] = realloc(result.C[i - iter_start], col * sizeof(int));
            }
            else {
              free(result.B[i - iter_start]);
              result.B[i - iter_start] = NULL;
              free(result.C[i - iter_start]);
              result.C[i - iter_start] = NULL;
            }
            result.rowLengths[i - iter_start] = col;
            //result.maxRowLength = col ? (col > result.maxRowLength) : result.maxRowLength;
        }

    }
    double end = omp_get_wtime();
    double timeSpent = (double)(end - start);
    printf("time spent = %10.6f\n", timeSpent);

    return result;
}

int write_file(CompressedMatrix m, const char* filenameB, const char* filenameC) {
  FILE* fp = fopen(filenameB, "w");
  FILE* fp2 = fopen(filenameC, "w");
  if (fp != NULL && fp2 != NULL) {
    for (int i = 0; i < m.numRow; ++i) {
      for (int j = 0; j < m.rowLengths[i]; ++j) {
        fprintf(fp, "%d", m.B[i][j]);
        fprintf(fp2, "%d", m.C[i][j]);
        if (j < m.rowLengths[i] - 1) {
          fprintf(fp, ",");
          fprintf(fp2, ",");
        }
      }
      fprintf(fp, "\n");
      fprintf(fp2, "\n");
    }
    fclose(fp);
    fclose(fp2);
    return 1;
  }
  return 0;
}

int mpi_independent_write(int rank, CompressedMatrix m, const char* filenameB, const char* filenameC) {
  MPI_File fileB;
  int err = MPI_File_open(MPI_COMM_WORLD, filenameB, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileB);
  MPI_File fileC;
  int err1 = MPI_File_open(MPI_COMM_WORLD, filenameC, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileC);
  if (err == MPI_SUCCESS && err1 == MPI_SUCCESS && rank == 0) {
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

void sample(int id, double probability, int rank, int numNode) {

  double startTime = omp_get_wtime();

  printf("sample%d:\n", id);
  MPI_Status status;
  int rowEachNode = ROW_NUM / numNode;
  CompressedMatrix X;
  initCompressedMatrix(&X, ROW_NUM);
  CompressedMatrix Y;
  initCompressedMatrix(&Y, ROW_NUM);
  if (rank == 0) {
    fill_compressed_matrix(&X, probability, id);
    fill_compressed_matrix(&Y, probability, id);
  }

  //send X and Y rowLengths to others
  printf("bcast rowlengths\n");
  MPI_Bcast(X.rowLengths, ROW_NUM, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(Y.rowLengths, ROW_NUM, MPI_INT, 0, MPI_COMM_WORLD);

  //according to rowLengths alloc Y.B, Y.C memory space
  if (rank != 0) {
    for (int i = 0; i < Y.numRow; ++i) {
      Y.B[i] = calloc(Y.rowLengths[i], sizeof(int));
      Y.C[i] = calloc(Y.rowLengths[i], sizeof(int));
    }
  }


  //send X
  printf("send X\n");
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
  printf("bcast Y\n");
  for (int i = 0; i < ROW_NUM; ++i) {
    MPI_Bcast(Y.B[i], Y.rowLengths[i], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y.C[i], Y.rowLengths[i], MPI_INT, 0, MPI_COMM_WORLD);
  }
    
  //openmpi matrix multiplication
  printf("start_omp_matrix_multiplication\n");
  CompressedMatrix result = omp_matrix_multiply(X, Y, g_threadsPerProc, rank, numNode);

  //recv result
  printf("recv result\n");
  CompressedMatrix finalResult;
  if (rank == 0) {
    initCompressedMatrix(&finalResult, ROW_NUM);
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
  if (g_writeToFile != 0) {
    printf("write to file\n");
    //mpi_independent_write(rank, X, "XB.txt", "XC.txt");
    //mpi_independent_write(rank, Y, "YB.txt", "YC.txt");
    //mpi_independent_write(rank, finalResult, "XYB.txt", "XYC.txt");
    if (rank == 0) {
      write_file(X, "XB.txt", "XC.txt");
      write_file(Y, "YB.txt", "YC.txt");
      write_file(finalResult, "XYB.txt", "XYC.txt");
    }
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

  //master thread have received all, so comput time consuming
  double endTime = omp_get_wtime();
  if (rank == 0) {
    printf("total_time_spent = %10.6f\n", endTime - startTime);
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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    g_numNodes = size;

    if (rank == 0) {
      printf("matrix_rows:%d\n", ROW_NUM);
      printf("mpi_numNodes:%d\n", size);
      printf("thread_per_proc:%d\n", g_threadsPerProc);
      printf("probability:%f\n", probability);
      printf("write_to_file:%d\n", g_writeToFile);
    }
    
    sample(0, probability, rank, size);

    MPI_Finalize();

    return 0;
}


