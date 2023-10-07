#include "tensor.h"
#include "utils.h"

#include <iostream>

int recursiveReduce(int *data, int const size) {
  // terminate check
  if (size == 1)
    return data[0];
  // renew the stride
  int const stride = size / 2;
  if (size % 2 == 1) {
    for (int i = 0; i < stride; i++) {
      data[i] += data[i + stride];
    }
    data[0] += data[size - 1];
  } else {
    for (int i = 0; i < stride; i++) {
      data[i] += data[i + stride];
    }
  }
  // call
  return recursiveReduce(data, stride);
}

__global__ void warmup(int *g_odata, int *g_idata, unsigned int n) {
  // set thread ID
  unsigned int tid = threadIdx.x;
  // boundary check
  if (tid >= n)
    return;
  // convert global data pointer to the
  int *idata = g_idata + blockIdx.x * blockDim.x;
  // in-place reduction in global memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      idata[tid] += idata[tid + stride];
    }
    // synchronize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighbored(int *g_odata, int *g_idata, size_t n) {
  unsigned int tid = threadIdx.x;
  // boundary check
  if (tid > n) {
    return;
  }
  int *idata = g_idata + blockDim.x * blockIdx.x;

  for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceNeighboredLess(int *g_odata, int *g_idata, size_t n) {
  unsigned int tid = threadIdx.x;
  // boundary check
  if (tid > n) {
    return;
  }
  int *idata = g_idata + blockDim.x * blockIdx.x;

  for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
    // key step
    unsigned int idx = 2 * stride * tid;

    if (idx < blockDim.x) {
      idata[idx] += idata[idx + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceInterleaved(int *g_odata, int *g_idata, size_t n) {
  unsigned int tid = threadIdx.x;
  // boundary check
  if (tid > n) {
    return;
  }
  int *idata = g_idata + blockDim.x * blockIdx.x;

  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnroll2(int *g_odata, int *g_idata, size_t n) {
  unsigned int tid = threadIdx.x;
  // boundary check
  if (tid >= n) {
    return;
  }

  unsigned int idx = blockDim.x * blockIdx.x * 2 + tid;
  if (idx + blockDim.x < n) {
    g_idata[idx] += g_idata[idx + blockDim.x];
  }
  __syncthreads();

  int *idata = g_idata + blockDim.x * blockIdx.x * 2;
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

class Test {
private:
  constexpr static int size_ = 1 << 24;
  constexpr static size_t bytes_ = size_ * sizeof(int);

  constexpr static int blocksize_ = 1024;
  constexpr static dim3 block_{blocksize_};
  constexpr static dim3 grid_{(size_ - 1) / block_.x + 1};

  int *idata_host_, *odata_host_;

  double iStart_, iElaps_;

public:
  Test() {
    printf("	with array size %d  ", size_);
    printf("grid %d block %d \n", grid_.x, block_.x);

    idata_host_ = (int *)malloc(bytes_);
    odata_host_ = (int *)malloc(grid_.x * sizeof(int));
    // initialize the array
    initialData_int(idata_host_, size_);
  }

  ~Test() {
    free(idata_host_);
    free(odata_host_);
  }

  void exec_cpu() {
    int *tmp = (int *)malloc(bytes_);
    memcpy(tmp, idata_host_, bytes_);

    int cpu_sum = 0;
    iStart_ = cpuSecond();
    // cpu_sum = recursiveReduce(tmp, size);
    for (int i = 0; i < size_; i++)
      cpu_sum += tmp[i];
    printf("cpu sum:%d \n", cpu_sum);
    iElaps_ = cpuSecond() - iStart_;
    printf("cpu reduce               elapsed %lf ms cpu_sum: %d\n", iElaps_,
           cpu_sum);

    free(tmp);
  }

  template <class Func>
  void exec_dev(Func func, std::string func_name, size_t unroll = 1) {
    mq::DeviceTensor<int> idata_dev(bytes_), odata_dev(grid_.x * sizeof(int));

    idata_dev.MemcpyHostToDevice(idata_host_, bytes_);
    CHECK(cudaDeviceSynchronize())

    iStart_ = cpuSecond();
    func<<<grid_.x / unroll, block_>>>(odata_dev.get(), idata_dev.get(), size_);
    cudaDeviceSynchronize();
    iElaps_ = cpuSecond() - iStart_;

    odata_dev.MemcpyDeviceToHost(odata_host_, grid_.x * sizeof(int));
    int gpu_sum = 0;
    for (int i = 0; i < grid_.x / unroll; i++)
      gpu_sum += odata_host_[i];

    printf("gpu %-*s elapsed %lf ms gpu_sum: %d <<<grid %zu "
           "block %d>>>\n",
           20, func_name.data(), iElaps_, gpu_sum, grid_.x / unroll, block_.x);
  }
};

int main(int argc, char **argv) {
  initDevice(0);

  Test test{};

  test.exec_cpu();
  test.exec_dev(warmup, "warmup");
  test.exec_dev(reduceNeighbored, "reduceNeighbored");
  test.exec_dev(reduceNeighboredLess, "reduceNeighboredLess");
  test.exec_dev(reduceInterleaved, "reduceInterleaved");
  test.exec_dev(reduceUnroll2, "reduceUnroll2", 2);
}
