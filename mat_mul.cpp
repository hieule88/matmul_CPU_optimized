#include "mat_mul.h"
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <pthread.h>
#include <immintrin.h>
#include <algorithm>
#include "util.h"
#define ITILESIZE (64)
#define JTILESIZE (512)
#define KTILESIZE (512)
static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;
int rows_per_process;
MPI_Request requestS, requestC;
void* mat_mul_thread(void *arg) {
  int tid = (long)arg;
  int is = rows_per_process / num_threads * tid + std::min(tid, rows_per_process % num_threads);
  int ie = rows_per_process / num_threads * (tid + 1) + std::min(tid + 1, rows_per_process % num_threads);
  if (mpi_rank!=0) zero_mat(&C[is * N], ie - is, N);
  float a0, a1, a2, a3, a4, a5;
  float b00, b01, b02, b03, b04, b05, b10, b11, b12, b13, b14, b15, \
        b20, b21, b22, b23, b24, b25, b30, b31, b32, b33, b34, b35, \
        b40, b41, b42, b43, b44, b45, b50, b51, b52, b53, b54, b55, \
        b60, b61, b62, b63, b64, b65, b70, b71, b72, b73, b74, b75;
  int min_k, min_i, min_j;
  int kk, ii, jj, k, i, j;
  for (kk = 0; kk < K; kk+= KTILESIZE) {
    min_k = std::min(kk + KTILESIZE, K);
    for (ii = is; ii < ie; ii+= ITILESIZE) {
      min_i = std::min(ii + ITILESIZE, ie);
      for (jj = 0; jj < N; jj+= JTILESIZE) {
        min_j = std::min(jj + JTILESIZE, N);
        for (k = kk; k + 5 < min_k; k+=6) {
          for (i = ii; i < min_i; i++) {
            a0 = A[i * K + (k + 0)];
            a1 = A[i * K + (k + 1)];
            a2 = A[i * K + (k + 2)];
            a3 = A[i * K + (k + 3)];
            a4 = A[i * K + (k + 4)];
            a5 = A[i * K + (k + 5)];
            for (j = jj; j + 7 < min_j; j+=8) {
              b00 = B[(k + 0) * N + j];
              b01 = B[(k + 1) * N + j];
              b02 = B[(k + 2) * N + j];
              b03 = B[(k + 3) * N + j];
              b04 = B[(k + 4) * N + j];
              b05 = B[(k + 5) * N + j];
              C[i * N + j] += a0 * b00;
              C[i * N + j] += a1 * b01;
              C[i * N + j] += a2 * b02;
              C[i * N + j] += a3 * b03;
              C[i * N + j] += a4 * b04;
              C[i * N + j] += a5 * b05;

              b10 = B[(k + 0) * N + j + 1];
              b11 = B[(k + 1) * N + j + 1];
              b12 = B[(k + 2) * N + j + 1];
              b13 = B[(k + 3) * N + j + 1];
              b14 = B[(k + 4) * N + j + 1];
              b15 = B[(k + 5) * N + j + 1];
              C[i * N + j + 1] += a0 * b10;
              C[i * N + j + 1] += a1 * b11;
              C[i * N + j + 1] += a2 * b12;
              C[i * N + j + 1] += a3 * b13;
              C[i * N + j + 1] += a4 * b14;
              C[i * N + j + 1] += a5 * b15;

              b20 = B[(k + 0) * N + j + 2];
              b21 = B[(k + 1) * N + j + 2];
              b22 = B[(k + 2) * N + j + 2];
              b23 = B[(k + 3) * N + j + 2];
              b24 = B[(k + 4) * N + j + 2];
              b25 = B[(k + 5) * N + j + 2];
              C[i * N + j + 2] += a0 * b20;
              C[i * N + j + 2] += a1 * b21;
              C[i * N + j + 2] += a2 * b22;
              C[i * N + j + 2] += a3 * b23;
              C[i * N + j + 2] += a4 * b24;
              C[i * N + j + 2] += a5 * b25;

              b30 = B[(k + 0) * N + j + 3];
              b31 = B[(k + 1) * N + j + 3];
              b32 = B[(k + 2) * N + j + 3];
              b33 = B[(k + 3) * N + j + 3];
              b34 = B[(k + 4) * N + j + 3];
              b35 = B[(k + 5) * N + j + 3];
              C[i * N + j + 3] += a0 * b30;
              C[i * N + j + 3] += a1 * b31;
              C[i * N + j + 3] += a2 * b32;
              C[i * N + j + 3] += a3 * b33;
              C[i * N + j + 3] += a4 * b34;
              C[i * N + j + 3] += a5 * b35;

              b40 = B[(k + 0) * N + j + 4];
              b41 = B[(k + 1) * N + j + 4];
              b42 = B[(k + 2) * N + j + 4];
              b43 = B[(k + 3) * N + j + 4];
              b44 = B[(k + 4) * N + j + 4];
              b45 = B[(k + 5) * N + j + 4];
              C[i * N + j + 4] += a0 * b40;
              C[i * N + j + 4] += a1 * b41;
              C[i * N + j + 4] += a2 * b42;
              C[i * N + j + 4] += a3 * b43;
              C[i * N + j + 4] += a4 * b44;
              C[i * N + j + 4] += a5 * b45;

              b50 = B[(k + 0) * N + j + 5];
              b51 = B[(k + 1) * N + j + 5];
              b52 = B[(k + 2) * N + j + 5];
              b53 = B[(k + 3) * N + j + 5];
              b54 = B[(k + 4) * N + j + 5];
              b55 = B[(k + 5) * N + j + 5];
              C[i * N + j + 5] += a0 * b50;
              C[i * N + j + 5] += a1 * b51;
              C[i * N + j + 5] += a2 * b52;
              C[i * N + j + 5] += a3 * b53;
              C[i * N + j + 5] += a4 * b54;
              C[i * N + j + 5] += a5 * b55;

              b60 = B[(k + 0) * N + j + 6];
              b61 = B[(k + 1) * N + j + 6];
              b62 = B[(k + 2) * N + j + 6];
              b63 = B[(k + 3) * N + j + 6];
              b64 = B[(k + 4) * N + j + 6];
              b65 = B[(k + 5) * N + j + 6];
              C[i * N + j + 6] += a0 * b60;
              C[i * N + j + 6] += a1 * b61;
              C[i * N + j + 6] += a2 * b62;
              C[i * N + j + 6] += a3 * b63;
              C[i * N + j + 6] += a4 * b64;
              C[i * N + j + 6] += a5 * b65;

              b70 = B[(k + 0) * N + j + 7];
              b71 = B[(k + 1) * N + j + 7];
              b72 = B[(k + 2) * N + j + 7];
              b73 = B[(k + 3) * N + j + 7];
              b74 = B[(k + 4) * N + j + 7];
              b75 = B[(k + 5) * N + j + 7];
              C[i * N + j + 7] += a0 * b70;
              C[i * N + j + 7] += a1 * b71;
              C[i * N + j + 7] += a2 * b72;
              C[i * N + j + 7] += a3 * b73;
              C[i * N + j + 7] += a4 * b74;
              C[i * N + j + 7] += a5 * b75;
            }
            for (; j < min_j; j++) {
              C[i * N + j] += A[i * K + (k+0)] * B[(k+0) * N + j];
              C[i * N + j] += A[i * K + (k+1)] * B[(k+1) * N + j];
              C[i * N + j] += A[i * K + (k+2)] * B[(k+2) * N + j];
              C[i * N + j] += A[i * K + (k+3)] * B[(k+3) * N + j];
              C[i * N + j] += A[i * K + (k+4)] * B[(k+4) * N + j];
              C[i * N + j] += A[i * K + (k+5)] * B[(k+5) * N + j];
            }
          }
        }
        for (; k < min_k; k++)
          for (i = ii; i < min_i; i++)
            for (j = jj; j < min_j; j++)
              C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
  return NULL;
}

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size) {
  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;
  rows_per_process = M / mpi_world_size;

  // Scatter A and Bcast B from root process to all processes
  MPI_Iscatter(A, rows_per_process * K, MPI_FLOAT, A, rows_per_process * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &requestS);
  MPI_Ibcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requestC);
  // FIXME: for now, only root process runs the matrix multiplication.
  pthread_t threads[num_threads];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  cpu_set_t cpus[num_threads];
  MPI_Wait(&requestS, MPI_STATUS_IGNORE);
  MPI_Wait(&requestC, MPI_STATUS_IGNORE);
  for (long i = 0; i < num_threads; i++) {
    CPU_ZERO(&cpus[i]); // Khởi tạo cpu_set_t to zero
    CPU_SET(i, &cpus[i]); // Choose thresh of certain core to make result stable. This will make each core eacho thresh
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&threads[i], &attr, mat_mul_thread, (void*)i);
  }
  for (int thread = 0; thread < num_threads; ++thread) {
    pthread_join(threads[thread], NULL);
  }
  MPI_Request requestG;
  MPI_Igather(C, rows_per_process * N, MPI_FLOAT, C, rows_per_process * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requestG);
}