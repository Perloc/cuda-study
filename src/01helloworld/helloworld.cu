#include <stdio.h>

__global__ void print(void) { printf("GPU: Hello World!\n"); }

int main() {
  printf("CPU: Hello World\n");
  print<<<1, 10>>>();
  cudaDeviceReset();

  return 0;
}
