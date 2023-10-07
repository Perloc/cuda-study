#include "utils.h"
#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace mq {

template <class T> class DeviceTensor {
private:
  T *dev_addr_;
  size_t length_;

public:
  DeviceTensor(size_t nBytes) : length_(nBytes) {
    CHECK(cudaMalloc((void **)&dev_addr_, nBytes))
  }

  ~DeviceTensor() { cudaFree(dev_addr_); }

  void MemcpyHostToDevice(T *host_addr, size_t nBytes) {
    if (nBytes > length_) {
      throw "Out of range!\n";
    }
    CHECK(cudaMemcpy(dev_addr_, host_addr, nBytes, cudaMemcpyHostToDevice))
  }
  void MemcpyDeviceToHost(T *host_addr, size_t nBytes) {
    if (nBytes > length_) {
      throw "Out of range!\n";
    }
    CHECK(cudaMemcpy(host_addr, dev_addr_, nBytes, cudaMemcpyDeviceToHost))
  }

  T *get() { return dev_addr_; }
  size_t getLength() { return length_; }
};

} // namespace mq
