#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SHMOO 1
#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(Body *p, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {

  int nBodies  = 150000;
  int priority = 0;
  if (argc > 1) nBodies  = atoi(argv[1]);
  if (argc > 2) priority = atoi(argv[2]);

  const float dt = 0.01f; // time step
  const int nIters = 20;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  Body *d_p = (Body*)d_buf;

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0;

  cudaStream_t stream1;
  cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, priority);
  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    cudaMemcpyAsync(d_buf, buf, bytes, cudaMemcpyHostToDevice, stream1);
    bodyForce<<<nBlocks, BLOCK_SIZE, 0, stream1>>>(d_p, dt, nBodies); // compute interbody forces
    cudaMemcpyAsync(buf, d_buf, bytes, cudaMemcpyDeviceToHost, stream1);
    cudaDeviceSynchronize();

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed;
    }
    // printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
  }
  double avgTime = totalTime / (double)(nIters-1);

  // printf("%d, %0.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
  free(buf);
  cudaFree(d_buf);
  cudaStreamDestroy(stream1);
}
