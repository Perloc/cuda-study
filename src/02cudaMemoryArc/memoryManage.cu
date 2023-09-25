#include "utils.h"

#include <cstdio>
#include <iostream>
#include <thread>

void vecSumHost(float *x, float *y, float *res, size_t nElem) {
  for (int i = 0; i < nElem; i += 4) {
    res[i] = x[i] + y[i];
    res[i + 1] = x[i + 1] + y[i + 1];
    res[i + 2] = x[i + 2] + y[i + 2];
    res[i + 3] = x[i + 3] + y[i + 3];
  }
}

__global__ void vecSumGpu(float *x, float *y, float *res) {
  int i = threadIdx.x;
  res[i] = x[i] + y[i];
}

template <class T> void print(T *p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    std::cout << p[i] << ' ';
  }
  std::cout << '\n';
}

int main() {
  int dev = 0;
  cudaSetDevice(dev);

  constexpr int nElem = 32;

  constexpr size_t nBytes = nElem * sizeof(float);
  float *x_h = (float *)malloc(nBytes);
  float *y_h = (float *)malloc(nBytes);
  float *res_h = (float *)malloc(nBytes);
  float *res_from_dev = (float *)malloc(nBytes);

  initialData(x_h, nElem);
  initialData(y_h, nElem);
  memset(res_h, 0, nElem);
  memset(res_from_dev, 0, nElem);

  float *x_d, *y_d, *res_d;
  CHECK(cudaMalloc((float **)&x_d, nBytes));
  CHECK(cudaMalloc((float **)&y_d, nBytes));
  CHECK(cudaMalloc((float **)&res_d, nBytes));

  CHECK(cudaMemcpy(x_d, x_h, nBytes, cudaMemcpyHostToDevice))
  CHECK(cudaMemcpy(y_d, y_h, nBytes, cudaMemcpyHostToDevice))

  dim3 block(nElem);
  dim3 grid(nElem / block.x);

  vecSumGpu<<<grid, block>>>(x_d, y_d, res_d);
  CHECK(cudaMemcpy(res_from_dev, res_d, nBytes, cudaMemcpyDeviceToHost));

  vecSumHost(x_h, y_h, res_h, nElem);
  checkResult(res_h, res_from_dev, nElem);

  print(x_h, nElem);
  print(y_h, nElem);
  print(res_h, nElem);
  print(res_from_dev, nElem);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(res_d);
  free(x_h);
  free(y_h);
  free(res_h);
  free(res_from_dev);

  return 0;
}
