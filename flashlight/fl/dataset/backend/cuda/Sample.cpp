#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <flashlight/fl/common/backend/cuda/CudaUtils.h>
#include <flashlight/fl/dataset/Sample.h>

#define DEFAULT_PINNED_MEM_SIZE 32 * 1024 * 1024

namespace fl {
Sample::Sample() : status_(Status::SELF_MANAGED) {
  FL_CUDA_CHECK(cudaStreamCreate(&stream_));
  validStream_ = true;
}

Sample::Sample(const af::array& array)
    : array_(array), status_(Status::MANAGED_BY_AF) {}

void Sample::copyFromHost(
    void* hostPtr,
    af::dim4 dims,
    size_t sizeInBytes,
    af::dtype type) {
  array_ = af::array();
  dims_ = dims;
  status_ = Status::SELF_MANAGED;
  type_ = type;
  sizeInBytes_ = sizeInBytes;

  if (sizeInBytes > sysMemSize_) {
    if (sysMemSize_ != 0) {
      FL_CUDA_CHECK(cudaFreeHost(sysMemPtr_));
      FL_CUDA_CHECK(cudaMallocHost(&sysMemPtr_, sizeInBytes));
      sysMemSize_ = sizeInBytes;
    } else {
      auto size = DEFAULT_PINNED_MEM_SIZE;
      size = size > sizeInBytes ? size : sizeInBytes;
      FL_CUDA_CHECK(cudaMallocHost(&sysMemPtr_, size));
      sysMemSize_ = size;
    }
  }

  memcpy(sysMemPtr_, hostPtr, sizeInBytes);
}

af::array&& Sample::array() {
  if (status_ == Status::RETIRED) {
    throw std::runtime_error("array() is called after the content is moved.");
  }

  if (status_ == Status::SELF_MANAGED) {
    if (!sysMemPtr_) {
      array_ = af::array();
    } else {
      if (type_ == u8) {
        array_ = af::array(dims_, (unsigned char*)sysMemPtr_);
      } else {
        std::string message =
            "Copy from arrayfire type: " + std::to_string((uint)type_) +
            ", is not implemented yet.";
        throw std::runtime_error(message);
      }
    }
  }else if (status_ == Status::IN_TRANSION_FROM_HOST_TO_DEVICE) {
    FL_CUDA_CHECK(cudaStreamSynchronize(stream_));
    array_.unlock();
  }

  status_ = Status::RETIRED;
  return std::move(array_);
}

af::dim4 Sample::dims() const {
  if (status_ == Status::MANAGED_BY_AF) {
    return array_.dims();
  }
  return dims_;
}

dim_t Sample::dims(unsigned int dim) const {
  if (status_ == Status::MANAGED_BY_AF) {
    return array_.dims(dim);
  }
  return dims_[dim];
}

dim_t Sample::elements() const {
  if (status_ == Status::MANAGED_BY_AF) {
    return array_.elements();
  }

  return dims_[0] * dims_[1] * dims_[2] * dims_[3];
}

void Sample::setArray(af::array& array) {
  status_ = Status::MANAGED_BY_AF;
  array_ = array;
}

void Sample::toDeviceAsync() {
  if (status_ != Status::SELF_MANAGED) {
    return;
  }

  if (!sysMemPtr_) {
    array_ = af::array();
    status_ = Status::MANAGED_BY_AF;
    return;
  }

  status_ = Status::IN_TRANSION_FROM_HOST_TO_DEVICE;
  array_ = af::array(dims_, type_);
  array_.eval();
  auto dArray = array_.device<void>();
  cudaMemcpyAsync(
      dArray, sysMemPtr_, sizeInBytes_, cudaMemcpyHostToDevice, stream_);
}

af::dtype Sample::type() const {
  if (status_ == Status::MANAGED_BY_AF) {
    return array_.type();
  }
  return type_;
}

Sample::~Sample() {
  if (sysMemPtr_) {
    FL_CUDA_CHECK(cudaFreeHost(sysMemPtr_));
  }
  if (validStream_) {
    FL_CUDA_CHECK(cudaStreamDestroy(stream_));
  }
}
} // namespace fl
