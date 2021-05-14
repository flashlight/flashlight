#pragma once

#include <memory>

#include <arrayfire.h>

#ifdef FL_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace fl {

class Sample {
 public:
  Sample();
  Sample(const af::array& array);

  /**
   * Resets the `Sample` and reload new data from the host memory. This method
   * copies the data to which `hostPtr` points to a non-pagable memory.
   * `hostPtr` can be freed immediately after this call.
   * @param[in] hostPtr Pointer to data on CPU memory.
   * @param[in] dims Dimension of the tensor to which `hostPtr` points.
   * @param[in] sizeInBytes Number of bytes to be copied.
   * @param[in] type Suitable Arrayfire type.
   */
  void copyFromHost(
      void* hostPtr,
      af::dim4 dims,
      size_t sizeInBytes,
      af::dtype type);

  /**
   * Returns an `af::array` that incorporates the data. If the data is on the
   * system memory and/or not managed by Arrayfire, it takes all the
   * necessary steps to create and array from the data and returns it.
   */
  af::array&& array();

  /**
   * Returns the dimensions of the Tensor that is wrapped by `Sample`.
   */
  af::dim4 dims() const;

  /**
   * Returns the dimensions of the Tensor that is wrapped by `Sample`.
   */
  dim_t dims(unsigned int dim) const;

  /**
   * Returns the number of elements that exist in the tensor that is wrapped by
   * `Sample`.
   */
  dim_t elements() const;

  void setArray(af::array& array);

  af::dtype type() const;

  /**
   * If the backend is `CUDA` and the data is in the system memory it initiates
   * an asynchronous host to device copy. Otherwise, it is a NOP. Calling
   * `toDeviceAsync` is safe and does not require any subsequent
   * synchronization even if the `array` method is immediately called after.
   */
  void toDeviceAsync();

  ~Sample();

 private:
  /**
   * A private structure that keeps track of the status.
   * `SELF_MANAGED`: Data is in the system memory and is managed by `Sample`
   * itself.
   * `MANAGED_BY_AF`: Data is in an `af::array` and is managed by
   * arrayfire. 
   * `IN_TRANSIT_SELF_TO_AF`: Data is being transferred to an
   * `af::array` in an asynchronous fashion. 
   * `RETIRED`: The content is moved.
   */
  enum Status {
    SELF_MANAGED = 0,
    MANAGED_BY_AF,
    IN_TRANSION_FROM_HOST_TO_DEVICE,
    RETIRED
  };

  Status status_;

  af::dim4 dims_;
  af::dtype type_;
  size_t sizeInBytes_;

  void* sysMemPtr_ = nullptr;
  size_t sysMemSize_ = 0;

  af::array array_;

#ifdef FL_USE_CUDA
  cudaStream_t stream_;
  bool validStream_ = false;
#endif
};

using SamplePtr = std::shared_ptr<Sample>;
} // namespace fl
