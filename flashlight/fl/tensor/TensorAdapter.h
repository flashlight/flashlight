/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Convert a Tensor from an adapter or via arguments for an adapter.
 *
 * @param[in] t the thing to convert
 * @return a Tensor containing the ArrayFire array
 */
template <typename Impl, typename... T>
Tensor toTensor(T&&... t) {
  return Tensor(std::make_unique<Impl>(std::forward<T>(t)...));
}

/**
 * The implementation interface for Flashlight Tensor backends.
 *
 * Defines the implementation requirements and behaviors for a particular tensor
 * backend. These implementations can be tested against default
 * backend/reference implementations.
 *
 * Derived types from TensorAdapterBase should not be constructed or operated on
 * literally. Calls to their member implementations are dispatched via visiting
 * Tensor or other interfaces.
 */
class FL_API TensorAdapterBase {
 public:
  TensorAdapterBase() = default;
  virtual ~TensorAdapterBase() = default;

  /**
   * Construct a tensor from some existing data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] type the type of the new tensor
   * @param[in] memoryLocation the location of the buffer
   */
  TensorAdapterBase(
      const Shape& shape,
      fl::dtype type,
      void* ptr,
      MemoryLocation memoryLocation);

  TensorAdapterBase(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  /**
   * Copies the tensor adapter. The copy is not required to be eager -- the
   * implementation can use copy-on-write.
   */
  virtual std::unique_ptr<TensorAdapterBase> clone() const = 0;

  /**
   * Gets the tensor's associated backend.
   *
   * @return TensorBackendType enum associated with the backend
   */
  virtual TensorBackendType backendType() const = 0;

  /**
   * Gets the backend for a tensor with this adapter implementation.
   *
   * @return the TensorBackend instance backing this particular tensor.
   */
  virtual TensorBackend& backend() const = 0;

  /**
   * Deep copy the tensor, including underlying data.
   */
  virtual Tensor copy() = 0;

  /**
   * Shallow copy the tensor - return a tensor that points to the same
   * underlying data.
   */
  virtual Tensor shallowCopy() = 0;

  /**
   * Get the shape of a tensor.
   *
   * @return the shape of the tensor
   */
  virtual const Shape& shape() = 0;

  /**
   * Get the data type of tensor.
   *
   * @return the dtype of the tensor
   */
  virtual dtype type() = 0;

  /**
   * Returns if the tensor is sparse.
   *
   * @return true if the tensor is sparse, else false
   */
  virtual bool isSparse() = 0;

  /**
   * Get a tensor's location, host or some device.
   *
   * @return the tensor's location
   */
  virtual Location location() = 0;

  /**
   * Populate a pointer with a scalar for the first element of the tensor.
   */
  virtual void scalar(void* out) = 0;

  /**
   * Returns a pointer to the tensor in device memory
   */
  virtual void device(void** out) = 0;

  /**
   * Populates a pointer with a pointer value in memory pointing to a host
   * buffer containing tensor data.
   */
  virtual void host(void* out) = 0;

  /**
   * Unlocks any device memory associated with the tensor that was acquired with
   * Tensor::device(), making it eligible to be freed.
   */
  virtual void unlock() = 0;

  /**
   * Returns true if the tensor has been memory-locked per a call to
   * Tensor::device<T>().
   *
   * @return true if the tensor is locked and a device pointer is active.
   */
  virtual bool isLocked() = 0;

  /**
   * Returns a bool based on Tensor contiguousness in memory.
   */
  virtual bool isContiguous() = 0;

  /**
   * Get the dimension-wise strides for this tensor - the number of bytes to
   * step in each direction when traversing.
   */
  virtual Shape strides() = 0;

  /**
   * Get the stream which contains(ed) the computation required to realize an
   * up-to-date value for this tensor. For instance, `device()` may not yield a
   * pointer to the up-to-date value -- to use this pointer, `Stream::sync` or
   * `Stream::relativeSync` is required.
   *
   * @return an immutable reference to the stream that contains(ed) the
   * computations which create this tensor.
   */
  virtual const Stream& stream() const = 0;

  /**
   * Returns a tensor with elements cast as a particular type
   *
   * @param[in] the type to which to cast the tensor
   * @return a tensor with element-wise cast to the new type
   */
  virtual Tensor astype(const dtype type) = 0;

  /**
   * Index into a tensor with a variable number of indices.
   *
   * @param[in] indices a vector of Index references
   * @return an indexed tensor
   */
  virtual Tensor index(const std::vector<Index>& indices) = 0;

  /**
   * Returns a representation of the tensor in 1 dimension.
   *
   * @return a 1D version of this tensor
   */
  virtual Tensor flatten() const = 0;

  /**
   * Returns a tensor indexed from this tensor but indexed as a 1D/flattened
   * tensor.
   *
   * @return a 1D version of this tensor 1D-indexed with the given index.
   */
  virtual Tensor flat(const Index& idx) const = 0;

  /**
   * Returns a copy of the tensor that is contiguous in memory.
   */
  virtual Tensor asContiguousTensor() = 0;

  /**
   * Sets arbitrary data on a tensor. May be a no-op for some backends.
   */
  virtual void setContext(void* context) = 0;

  /**
   * Sets arbitrary data on a tensor. May be a no-op for some backends.
   *
   * @return An arbitrary payload
   */
  virtual void* getContext() = 0;

  /**
   * Return a string representation of a Tensor. Not intended to be portable
   * across backends.
   */
  virtual std::string toString() = 0;

  /**
   * Write a string representation of a tensor to an output stream.
   */
  virtual std::ostream& operator<<(std::ostream& ostr) = 0;

  /******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(OP, TYPE) virtual void OP(const TYPE& val) = 0;
#define ASSIGN_OP(OP)                 \
  ASSIGN_OP_TYPE(OP, Tensor);         \
  ASSIGN_OP_TYPE(OP, double);         \
  ASSIGN_OP_TYPE(OP, float);          \
  ASSIGN_OP_TYPE(OP, int);            \
  ASSIGN_OP_TYPE(OP, unsigned);       \
  ASSIGN_OP_TYPE(OP, bool);           \
  ASSIGN_OP_TYPE(OP, char);           \
  ASSIGN_OP_TYPE(OP, unsigned char);  \
  ASSIGN_OP_TYPE(OP, short);          \
  ASSIGN_OP_TYPE(OP, unsigned short); \
  ASSIGN_OP_TYPE(OP, long);           \
  ASSIGN_OP_TYPE(OP, unsigned long);  \
  ASSIGN_OP_TYPE(OP, long long);      \
  ASSIGN_OP_TYPE(OP, unsigned long long);

  ASSIGN_OP(assign); // =
  ASSIGN_OP(inPlaceAdd); // +=
  ASSIGN_OP(inPlaceSubtract); // -=
  ASSIGN_OP(inPlaceMultiply); // *=
  ASSIGN_OP(inPlaceDivide); // /=
#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP
};

namespace detail {

/*
 * An interface with which to construct a tensor. Templated based on used tensor
 * adapters.
 */
struct FL_API TensorCreator {
  virtual ~TensorCreator() = default;

  // General tensor ctor
  virtual std::unique_ptr<TensorAdapterBase> get(
      const Shape& shape = {0}, // 0 shape is an empty Tensor
      fl::dtype type = fl::dtype::f32,
      const void* ptr = nullptr,
      MemoryLocation memoryLocation = MemoryLocation::Host) const = 0;

  // Sparse tensor ctor
  virtual std::unique_ptr<TensorAdapterBase> get(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType) const = 0;
};

template <typename T>
struct TensorCreatorImpl : public TensorCreator {
  TensorCreatorImpl() = default;
  ~TensorCreatorImpl() override = default;

  std::unique_ptr<TensorAdapterBase> get(
      const Shape& shape = {0}, // 0 shape is an empty Tensor
      fl::dtype type = fl::dtype::f32,
      const void* ptr = nullptr,
      MemoryLocation memoryLocation = MemoryLocation::Host) const override {
    return std::make_unique<T>(shape, type, ptr, memoryLocation);
  }

  std::unique_ptr<TensorAdapterBase> get(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType) const override {
    return std::make_unique<T>(
        nRows, nCols, values, rowIdx, colIdx, storageType);
  }
};

/*
 * A singleton to hold a closure which creates a new tensor of default type. For
 * internal use only - use setDefaultTensorType<T>() to set the type with which
 * to create a default tensor.
 */
class FL_API DefaultTensorType {
  // The function to use to create a tensor of default type.
  std::unique_ptr<TensorCreator> creationFunc_;

 public:
  static DefaultTensorType& getInstance();
  DefaultTensorType();

  std::unique_ptr<TensorCreator> swap(
      std::unique_ptr<TensorCreator> creator) noexcept;
  const TensorCreator& getTensorCreator() const;

  DefaultTensorType(DefaultTensorType const&) = delete;
  void operator=(DefaultTensorType const&) = delete;
};

/**
 * Get an instance of the default tensor adapter.
 */
template <typename... T>
std::unique_ptr<TensorAdapterBase> getDefaultAdapter(T&&... t) {
  return DefaultTensorType::getInstance().getTensorCreator().get(
      std::forward<T>(t)...);
}

} // namespace detail

/**
 * Set the default tensor type for which new tensors will be created.
 *
 * This function is parameterized by the tensor type. Usage is as follows:
 * \code
   fl::setDefaultTensorType<TensorType>()
   \endcode
 *
 * Where TensorType is derived from TensorAdapterBase.
 */
template <typename T>
void setDefaultTensorType() {
  static_assert(
      std::is_base_of<TensorAdapterBase, T>::value,
      "setDefaultTensorType: T must be a derived type of TensorAdapterBase");
  fl::detail::DefaultTensorType::getInstance().swap(
      std::make_unique<detail::TensorCreatorImpl<T>>());
}

template <typename T, typename B>
void withTensorType(B func) {
  static_assert(
      std::is_base_of<TensorAdapterBase, T>::value,
      "withTensorType: T must be a derived type of TensorAdapterBase");

  // Swap
  auto oldCreator = fl::detail::DefaultTensorType::getInstance().swap(
      std::make_unique<detail::TensorCreatorImpl<T>>());
  func();
  // Restore
  fl::detail::DefaultTensorType::getInstance().swap(std::move(oldCreator));
}

} // namespace fl
