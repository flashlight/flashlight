/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <utility>

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
class TensorAdapterBase {
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

  /**
   * Copies the tensor adapter. The implementation defines whether or not tensor
   * data itself is copied - this is not an implementation requirement.
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
  virtual void host(void** out) = 0;

  /**
   * Unlocks any device memory associated with the tensor that was acquired with
   * Tensor::device(), making it eligible to be freed.
   */
  virtual void unlock() = 0;

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
   * Sets arbitrary data on a tensor. May be a no-op for some backends.
   */
  virtual void setContext(void* context) = 0;

  /**
   * Sets arbitrary data on a tensor. May be a no-op for some backends.
   *
   * @return An arbitrary payload
   */
  virtual void* getContext() = 0;

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

using DefaultTensorTypeFunc_t =
    std::function<std::unique_ptr<TensorAdapterBase>(
        const Shape& shape,
        fl::dtype type,
        void* ptr,
        MemoryLocation memoryLocation)>;

/*
 * A singleton to hold a closure which creates a new tensor of default type. For
 * internal use only - use setDefaultTensorType<T>() to set the type with which
 * to create a default tensor.
 */
class DefaultTensorType {
  // The function to use to create a tensor of default type.
  DefaultTensorTypeFunc_t creationFunc_;

 public:
  static DefaultTensorType& getInstance();
  DefaultTensorType();

  void setCreationFunc(DefaultTensorTypeFunc_t&& func);
  const DefaultTensorTypeFunc_t& getCreationFunc() const;

  DefaultTensorType(DefaultTensorType const&) = delete;
  void operator=(DefaultTensorType const&) = delete;
};

/**
 * Get an instance of the default tensor adapter.
 */
std::unique_ptr<TensorAdapterBase> getDefaultAdapter(
    const Shape& shape = Shape(),
    fl::dtype type = fl::dtype::f32,
    void* ptr = nullptr,
    MemoryLocation memoryLocation = MemoryLocation::Host);

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

  fl::detail::DefaultTensorType::getInstance().setCreationFunc(
      [](const Shape& shape,
         fl::dtype type,
         void* ptr,
         MemoryLocation memoryLocation) {
        return std::make_unique<T>(shape, type, ptr, memoryLocation);
      });
}

template <typename T, typename B>
void withTensorType(B func) {
  // Save for later. Copy
  auto oldCreationFunc =
      fl::detail::DefaultTensorType::getInstance().getCreationFunc();
  // Set new tensor type and execute
  fl::setDefaultTensorType<T>();
  func();
  // Restore old func
  fl::detail::DefaultTensorType::getInstance().setCreationFunc(
      std::move(oldCreationFunc));
}

} // namespace fl
