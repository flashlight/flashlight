/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

class Tensor;

/**
 * \defgroup tensor_constants Tensor constants
 * @{
 */

/// Enum for various tensor backends.
enum class TensorBackendType { Stub, Tracer, ArrayFire, OneDnn, Jit };

// See TensorAdapter.h
class TensorAdapterBase;

// See TensorBackend.h
class TensorBackend;

// See Index.h
struct Index;

// See runtime/Stream.h
class Stream;

/// Location of memory or tensors.
enum class Location { Host, Device };
/// Alias to make it semantically clearer when referring to buffer location
using MemoryLocation = Location;

/// Tensor storage types.
enum class StorageType { Dense = 0, CSR = 1, CSC = 2, COO = 3 };

/* @} */

namespace detail {

FL_API std::unique_ptr<TensorAdapterBase> releaseAdapter(Tensor&& t);
FL_API std::unique_ptr<TensorAdapterBase> releaseAdapterUnsafe(Tensor& t);

} // namespace detail

/**
 * A Tensor on which computation can be performed.
 *
 * Underlying implementations of tensor operations are implemented via types
 * derived from `TensorAdapterBase`; these implementations also store state
 * associated with the tensor. Tensor stores a pointer to the implementation,
 * which can be swapped out if the implementation of backend
 * changes.`TensorAdapterBase` implementations may differ across tensor
 * libraries, hardware-specific libraries or compilers and DSLs.
 *
 * \todo: Improve documentation throughout.
 *
 * \warning This API may break and is not yet stable.
 */
class FL_API Tensor {
  // The tensor adapter for the tensor
  std::unique_ptr<TensorAdapterBase> impl_;

  /*
   * Construct a tensor with a given shape and using an existing buffer.
   *
   * This constructor is a void* specialization to facilitate interface
   * compliance with TensorAdapter and is intentionally private.
   */
  Tensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      MemoryLocation memoryLocation);

  /**
   * Shallow-copies the tensor, returning a tensor that points to the same
   * underlying data.
   *
   * For internal use only. Tensor implementations should define when and where
   * deep copies happen based on their dataflow graph abstractions.
   *
   * \todo slated for removal. Rely on copy-on-write and fix bad refcount
   * issues.
   */
  Tensor shallowCopy() const;
  // shallowCopy() above is used in DevicePtr given that it doesn't mutate
  // tensors in place with tensor operations, and only pulls out memory.
  friend class DevicePtr;
  // also used in tensor abstractions that wrap and call tensor ops:
  friend class TracerTensorBase;

  /**
   * Release and transfer ownership of the tensor's underlying
   * TensorAdapterBase.
   *
   * NB: After unlocking the adapter, the resulting Tensor should
   * *probably* be destroyed, as it has no adapter and thus can't perform any
   * operations.
   */

  std::unique_ptr<TensorAdapterBase> releaseAdapter();
  friend std::unique_ptr<TensorAdapterBase> detail::releaseAdapter(Tensor&& t);
  friend std::unique_ptr<TensorAdapterBase> detail::releaseAdapterUnsafe(
      Tensor& t);

 public:
  explicit Tensor(std::unique_ptr<TensorAdapterBase> adapter);
  virtual ~Tensor();

  /**
   * Copy constructor - calls the implementation-defined copy constructor for
   * the TensorAdapter.
   */
  Tensor(const Tensor& tensor);

  /**
   * Move constructor - moves the pointer to the TensorAdapter - performs no
   * other operations.
   */
  Tensor(Tensor&& tensor) noexcept;

  /**
   * Construct an empty tensor with the default tensor backend's tensor adapter.
   */
  Tensor();

  /**
   * Construct a tensor of a given shape (and optionally type) without
   * populating its data.
   *
   * @param[in] shape the shape of the tensor
   * @param[in] type (optional) the type of the tensor
   */
  explicit Tensor(const Shape& shape, fl::dtype type = fl::dtype::f32);

  /**
   * Construct an empty tensor of a given type.
   *
   * @param[in] type (optional) the type of the tensor
   */
  explicit Tensor(fl::dtype type);

  /**
   * Construct a sparse tensor.
   *
   * @param[in] nRows the number of rows of the tensor
   * @param[in] nCols the number of columns of the tensor
   * @param[in] values the values associated with the tensor
   * @param[in] rowIdx the row indices of the sparse array
   * @param[in] colIdx the the column indices of the sparse array
   * @param[in] storageType the storage type of the underlying tensor
   *
   * \todo Expand this API with getters as needed.
   */
  Tensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  /**
   * Create a tensor from a vector of values.
   *
   * @param[in] s the shape of the resulting tensor.
   * @param[in] v values with which to populate the tensor.
   * @return a tensor with values and shape as given.
   */
  template <typename T>
  static Tensor fromVector(Shape s, std::vector<T> v) {
    return Tensor(s, fl::dtype_traits<T>::fl_type, v.data(), Location::Host);
  }

  template <typename T, std::size_t S>
  static Tensor fromArray(Shape s, std::array<T, S> a) {
    return Tensor(s, fl::dtype_traits<T>::fl_type, a.data(), Location::Host);
  }

  template <typename T>
  static Tensor fromVector(Shape s, std::vector<T> v, dtype type) {
    return Tensor(s, type, v.data(), Location::Host);
  }

  template <typename T, std::size_t S>
  static Tensor fromArray(Shape s, std::array<T, S> a, dtype type) {
    return Tensor(s, type, a.data(), Location::Host);
  }

  template <typename T>
  static Tensor fromVector(std::vector<T> v) {
    return Tensor(
        {static_cast<long long>(v.size())},
        fl::dtype_traits<T>::fl_type,
        v.data(),
        Location::Host);
  }

  template <typename T, std::size_t S>
  static Tensor fromArray(std::array<T, S> a) {
    return Tensor(
        {static_cast<long long>(a.size())},
        fl::dtype_traits<T>::fl_type,
        a.data(),
        Location::Host);
  }

  /**
   * Create a tensor from an existing buffer.
   *
   * @param[in] s the shape of the resulting tensor.
   * @param[in] ptr the buffer containing the data
   * @param[in] memoryLocation the location in memory where the input buffer
   * with which to create the tensor resides.
   * @return a tensor with values and shape as given.
   */
  template <typename T>
  static Tensor fromBuffer(Shape s, const T* ptr, Location memoryLocation) {
    return Tensor(s, fl::dtype_traits<T>::fl_type, ptr, memoryLocation);
  }

  /**
   * Create a tensor from an existing byte buffer given a type.
   *
   * @param[in] s the shape of the resulting tensor.
   * @param[in] t the type of the underlying tensor
   * @param[in] ptr the buffer of bytes containing the data
   * @param[in] memoryLocation the location in memory where the input buffer
   * with which to create the tensor resides.
   * @return a tensor with values and shape as given.
   */
  static Tensor fromBuffer(
      Shape s,
      fl::dtype t,
      const uint8_t* ptr,
      Location memoryLocation) {
    return Tensor(s, t, ptr, memoryLocation);
  }

  /**
   * Deep-copies the tensor, including underlying data.
   */
  Tensor copy() const;

  /**
   * Get the shape of a tensor.
   *
   * @return the shape of the tensor
   */
  const Shape& shape() const;

  /**
   * Get a tensor's location, host or some device.
   *
   * @return the tensor's location
   */
  Location location() const;

  /**
   * Get the number of elements in the tensor.
   *
   * @return the size of the tensor in elements.
   */
  size_t elements() const;

  /**
   * Get the size of a given dimension of a tensor in the number of arguments.
   * Throws if the given dimension is larger than the number of tensor
   * dimensions.
   *
   * @return the number of elements at the given dimension
   */
  Dim dim(const size_t dim) const;

  /**
   * Get the number of directions of the tensor.
   *
   * @return the number of dimensions
   */
  int ndim() const;

  /**
   * Returns true if the tensor has zero elements, else false.
   *
   * @return true if the tensor is empty
   */
  bool isEmpty() const;

  /**
   * Returns true if the tensor has an associated underlying adapter.
   *
   * @return true if the tensor has a valid adapter
   */
  bool hasAdapter() const;

  /**
   * Get the tensor size in bytes.
   *
   * @return the size of the tensor in bytes.
   */
  size_t bytes() const;

  /**
   * Get the data type of tensor.
   *
   * @return the dtype of the tensor
   */
  dtype type() const;

  /**
   * Returns whether or not the tensor is sparse.
   *
   * @return true if the tensor is a sparse tensor, else false
   */
  bool isSparse() const;

  /**
   * Get this tensor's strides - the number of elements/coefficients to step
   * when moving along each dimension when traversing the tensor.
   *
   * @return a Shape containing strides in each dimension.
   */
  Shape strides() const;

  /**
   * Get the stream which contains(ed) the computation required to realize an
   * up-to-date value for this tensor. For instance, `device()` may not yield a
   * pointer to the up-to-date value -- to use this pointer, `Stream::sync` or
   * `Stream::relativeSync` is required.
   *
   * @return an immutable reference to the stream that contains(ed) the
   * computations which create this tensor.
   */
  virtual const Stream& stream() const;

  /**
   * Returns a tensor with elements cast as a particular type
   *
   * @param[in] type the type to which to cast the tensor
   * @return a tensor with element-wise cast to the new type
   */
  Tensor astype(const dtype type) const;

  /**
   * Index into a tensor using a vector of fl::Index references.
   *
   * @param[in] indices a vector of fl::Index references with which to index.
   * @return an indexed tensor
   */
  Tensor operator()(const std::vector<Index>& indices) const;

  /**
   * Index into a tensor using a variable number of fl::Index.
   *
   * @param[in] args fl::Index instances to use
   * @return an indexed tensor
   */
  template <typename... Ts>
  Tensor operator()(const Ts&... args) const {
    // TODO: add this back if acceptable with C++ 17 ABIs and a nvcc
    // static_assert(
    //     std::conjunction<std::is_constructible<Index, Ts>...>::value,
    //     "Tensor index operator can only take Index-compatible types - "
    //     "fl::range, fl::Tensor, fl::span, and integer types.");
    std::vector<Index> indices{{args...}};
    return this->operator()(indices);
  }

  /**
   * Returns a representation of the tensor in 1 dimension.
   *
   * @return a 1D version of this tensor 1D-indexed with the given index.
   */
  Tensor flatten() const;

  /**
   * Returns a tensor indexed from this tensor but indexed as a 1D/flattened
   * tensor.
   *
   * @return an indexed, 1D version of this tensor.
   */
  Tensor flat(const Index& idx) const;

  /**
   * Return a copy (depending on copy-on-write behavior of the underlying
   * implementation) of this tensor that is contigous in memory.
   *
   * @return an identical tensor that is contiguous in memory
   */
  Tensor asContiguousTensor() const;

  /**
   * Gets the backend enum from the underlying TensorAdapter.
   *
   * @return the backend in question
   */
  TensorBackendType backendType() const;

  /**
   * Gets the underlying tensor adapter implementation.
   *
   * @return the tensor adapter.
   */
  template <typename T>
  T& getAdapter() const {
    return *static_cast<T*>(impl_.get());
  }

  /**
   * Return the TensorBackend associated with this tensor.
   *
   * @return a TensorBackend.
   */
  TensorBackend& backend() const;

  /**
   * Return a scalar of a specified type for the tensor. If the tensor has more
   * than one element, returns the first element as a scalar.
   *
   * Throws an exception if the specified type does not match the dtype trait of
   * the underlying tensor. To implicitly cast the scalar regardless of the
   * underlying Tensor's dtype, use `asScalar`.
   *
   * @return a scalar of the first element in the tensor.
   */
  template <typename T>
  T scalar() const;

  /**
   * Return a scalar of the specified type of the tensor. If the specified type
   * does not match the tensor's underlying dtype, the scalar value is
   * implicitly cast.
   *
   * @return a scalar of the first element in the tensor cast to the specified
   * type.
   */
  template <typename T>
  T asScalar() const {
    // Implicitly cast to the requested return type
    switch (type()) {
      case dtype::f16:
        return astype(dtype::f32).scalar<float>();
      case dtype::f32:
        return scalar<float>();
      case dtype::f64:
        return scalar<double>();
      case dtype::s32:
        return scalar<int>();
      case dtype::u32:
        return scalar<unsigned int>();
      case dtype::b8:
        return scalar<char>();
      case dtype::u8:
        return scalar<unsigned char>();
      case dtype::s64:
        return scalar<long long>();
      case dtype::u64:
        return scalar<unsigned long long>();
      case dtype::s16:
        return scalar<short>();
      case dtype::u16:
        return scalar<unsigned short>();
      default:
        throw std::invalid_argument(
            "Tensor::asScaler - no castable type exists.");
    }
  }

  /**
   * Return a pointer to the tensor's underlying data per a certain type. This
   * pointer exists on the computation device.
   *
   * \note The memory allocated here will not be freed until Tensor:unlock() is
   * called.
   *
   * @return the requested pointer on the device.
   */
  template <typename T>
  T* device() const;

  /**
   * Populate a pointer value with the address of a Tensor's underlying buffer
   * on the computation device.
   *
   * \note The memory allocated here will not be freed until Tensor:unlock() is
   * called.
   *
   * @param[in] ptr the pointer to populate with the Tensor's buffer location on
   * device.
   */
  template <typename T>
  void device(T** ptr) const;

  /**
   * Returns a pointer to the tensor's underlying data, but on the host. If the
   * tensor is located on a device, makes a copy of device memory and returns a
   * buffer on the host containing the relevant memory.
   *
   * @return the requested pointer on the host.
   */
  template <typename T>
  T* host() const;

  /**
   * Populates an existing buffer with the tensor's underlying data, but on the
   * host. If the tensor is located on a device, makes a copy of device memory
   * and returns a buffer on the host containing the relevant memory.
   *
   * @param[in] ptr a pointer to the region of memory to populate with tensor
   * values
   */
  template <typename T>
  void host(T* ptr) const;

  /**
   * Returns a vector on the host contaning a flat representation of the tensor.
   * The resulting vector is a copy of the underlying tensor memory, even if on
   * the host.
   *
   * @return a vector in host memory containing
   */
  template <typename T>
  std::vector<T> toHostVector() const {
    if (isEmpty()) {
      return std::vector<T>();
    }
    std::vector<T> vec(this->elements());
    host(vec.data());
    return vec;
  }

  /**
   * Unlocks any device memory associated with the tensor that was acquired with
   * Tensor::device(), making it eligible to be freed.
   */
  void unlock() const;

  /**
   * Returns true if the tensor has been memory-locked per a call to
   * Tensor::device<T>(). After unlocking via Tensor::unlock(), the tensor is no
   * longer locked.
   *
   * @return true if the tensor is locked and a device pointer is active.
   */
  bool isLocked() const;

  /**
   * Returns if the Tensor is contiguous in its memory-based representation.
   *
   * @return a bool denoting Tensor contiguousness
   */
  bool isContiguous() const;

  /**
   * Stores arbitrary data on a tensor. For internal use/benchmarking only. This
   * may be a no-op for some backends.
   *
   * @param[in] data a pointer to arbitrary data to pass to a tensor impl.
   */
  void setContext(void* data);

  /**
   * Gets arbitrary data stored on a tensor. For internal use/benchmarking only.
   * This may be a no-op for some backends.
   *
   * @return a pointer to some implementation-defined data, else nullptr if a
   * no-op.
   */
  void* getContext() const;

  /**
   * Returns a string representation of a Tensor.
   *
   * \note This is backend-dependent. See Flashlight's serialization utilities
   * for ways to serialize Tensors that are portable across Tensor backends.
   *
   * @return a string representation of the Tensor.
   */
  std::string toString() const;

  /**
   * Write a string representation of a tensor to an output stream.
   */
  std::ostream& operator<<(std::ostream& ostr) const;

  /******************** Assignment Operators ********************/
#define ASSIGN_TENSOR_OP(OP) Tensor& OP(const Tensor& val);
#define ASSIGN_SCALAR_OP(OP)             \
  Tensor& OP(const double& val);         \
  Tensor& OP(const float& val);          \
  Tensor& OP(const int& val);            \
  Tensor& OP(const unsigned& val);       \
  Tensor& OP(const bool& val);           \
  Tensor& OP(const char& val);           \
  Tensor& OP(const unsigned char& val);  \
  Tensor& OP(const short& val);          \
  Tensor& OP(const unsigned short& val); \
  Tensor& OP(const long& val);           \
  Tensor& OP(const unsigned long& val);  \
  Tensor& OP(const long long& val);      \
  Tensor& OP(const unsigned long long& val);
#define ASSIGN_OP(OP)   \
  ASSIGN_TENSOR_OP(OP); \
  ASSIGN_SCALAR_OP(OP);

  ASSIGN_SCALAR_OP(operator=);
  ASSIGN_OP(operator+=);
  ASSIGN_OP(operator-=);
  ASSIGN_OP(operator*=);
  ASSIGN_OP(operator/=);
#undef ASSIGN_TENSOR_OP
#undef ASSIGN_SCALAR_OP
#undef ASSIGN_OP

  /* The following assignment operator differentiation via member method
   * ref-qualifier ensures that
   * 1. For `x = ...;`, the behavior is the same as the copy/move constructor.
   * 2. For `... = ...`, a copy is made from the rhs tensor data to the lhs one.
   *    This allows tensor mutation via indexing, e.g., `t(0, 0) = 42`.
   */
  Tensor& operator=(Tensor&& other) &;
  Tensor& operator=(Tensor&& other) &&;
  Tensor& operator=(const Tensor& other) &;
  Tensor& operator=(const Tensor& other) &&;
};

/**
 * \defgroup tensor_functions Tensor functions
 * @{
 */

/******************** Tensor Creation Functions ********************/

/**
 * Creates a new scalar Tensor with a particular value. Scalar tensors have an
 * empty Shape and 1 element by definition.
 *
 * @param[in] val the value with which to fill the tensor
 * @param[in] type the type of the tensor to create. Defaults to a value based
 * on the value type
 * @return a tensor of the specified shape filled with the specified value
 */
template <typename T>
FL_API Tensor
fromScalar(const T& val, const dtype type = dtype_traits<T>::ctype);

/**
 * Creates a new Tensor with a given Shape and filled with a particular value.
 *
 * @param[in] dims the shape of the tensor to create
 * @param[in] val the value with which to fill the tensor
 * @param[in] type the type of the tensor to create. Defaults to a value based
 * on the value type
 * @return a tensor of the specified shape filled with the specified value
 */
template <typename T>
FL_API Tensor full(
    const Shape& dims,
    const T& val,
    const dtype type = dtype_traits<T>::ctype);

/**
 * Return a the identity tensor of a given size and type.
 *
 * @param[in] dim the size of the dimension of the matrix (dim x dim)
 * @param[in] type the type of the resulting matrix
 */
FL_API Tensor identity(const Dim dim, const dtype type = dtype::f32);

/**
 * Return evenly-spaced values in a given interval. Generate values in the
 * interval `[start, stop)` steppping each element by the passed step.
 *
 * @param[in] start the start of the range
 * @param[in] end the end of the range, inclusive
 * @param[in] step the increment for each consecutive value in the range
 * @param[in] type the dtype of the resulting tensor
 *
 * @return a tensor containing the evenly-spaced values
 */
template <typename T>
FL_API Tensor arange(
    const T& start,
    const T& end,
    const T& step = 1,
    const dtype type = dtype_traits<T>::ctype);

/**
 * Create a tensor with [0, N] values along dimension given by seqDim and
 * tiled along the other dimensions. N is equal to the size of the shape along
 * the seqDim dimension.
 *
 * @param[in] shape the shape of the output tensor.
 * @param[in] seqDim (optional) the dimension along which the sequence increases
 * @param[in] type (optional) the dtype of the resulting tensor
 *
 * @return a tensor with the given shape with the sequence along the given
 * dimension, tiled along other dimensions.
 */
FL_API Tensor
arange(const Shape& shape, const Dim seqDim = 0, const dtype type = dtype::f32);

/**
 * Creates a sequence with the range `[0, dims.elements())` sequentially in the
 * shape given by dims, then tiles the result along the specified tile
 * dimensions.
 *
 * \todo an optimized version of this function is implemented only with the
 * ArrayFire backend.
 *
 * @param[in] dims the dimensions of the range
 * @param[in] tileDims the dimensions along which to tile
 * @param[in] type the dtype of the resulting tensoe
 *
 * @return
 */
FL_API Tensor iota(
    const Shape& dims,
    const Shape& tileDims = {1},
    const dtype type = dtype::f32);

/************************ Shaping and Indexing *************************/

/**
 * Change a tensor's shape without changing its underlying data.
 *
 * @param[in] tensor the tensor to reshape
 * @param[in] shape the new shape for the tensor
 * @return the reshaped tensor
 */
FL_API Tensor reshape(const Tensor& tensor, const Shape& shape);

/**
 * Permute the axes of a tensor. If no arguments are given, reverses the axes of
 * a tensor.
 *
 * @param[in] tensor the tensor to transpose
 * @param[in] axes (optional) the permuted indices of the tensor the kth access
 * of the output tensor will correspond to dims[k] in the input tensor. If this
 * argument is not passed, the axes of the input tensor will be reversed.
 * @return the permuted tensor
 */
FL_API Tensor transpose(const Tensor& tensor, const Shape& axes = {});

/**
 * Repeat the contents of a tensor a given number of times along specified
 * dimensions.
 *
 * @param[in] tensor the tensor to tile
 * @param[in] shape the number of times, along each dimension, which to tile the
 * tensor
 * @return the tiled tensor
 */
FL_API Tensor tile(const Tensor& tensor, const Shape& shape);

/**
 * Join or concatenate tensors together along a particular axis.
 *
 * @param[in] tensors a vector of tensors to concatenate
 * @param[in] axis the axis along which to concatenate tensors
 * @return a concatenated tensor
 */
FL_API Tensor
concatenate(const std::vector<Tensor>& tensors, const unsigned axis = 0);

/**
 * Join or concatenate tensors together along a particular axis.
 *
 * @param[in] axis the axis along which to concatenate tensors
 * @param[in] args tensors to concatenate
 * @return a concatenated tensor
 */
template <typename... Ts>
Tensor concatenate(unsigned axis, const Ts&... args) {
  std::vector<Tensor> tensors{{args...}};
  return concatenate(tensors, axis);
}

/**
 * Return the indices of elements that are non-zero. Indices correspond to a
 * flattened version of the input tensor.
 *
 * @param[in] tensor input tensor
 * @return a tensor containing the indices of the nonzero elements
 */
FL_API Tensor nonzero(const Tensor& tensor);

/// Padding types for the pad operator.
enum class PadType {
  /// pad with a constant zero value.
  Constant,
  /// pad with the values at the edges of the tensor
  Edge,
  /// pad with a reflection of the tensor mirrored along each edge
  Symmetric
};

/**
 * Pad a tensor with zeros.
 *
 * @param[in] input the input tensor to pad
 * @param[in] padWidths a vector of tuples representing padding (before, after)
 * tuples for each axis
 * @param[in] type the padding mode with which to pad the tensor - see `PadType`
 *
 * @return the padded tensor
 */
FL_API Tensor
pad(const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type = PadType::Constant);

/************************** Unary Operators ***************************/
/**
 * Element-wise negation of a tensor.
 *
 * @param[in] tensor the input tensor to negate.
 * @return a tensor with elements negated.
 */
FL_API Tensor negative(const Tensor& tensor);
inline Tensor operator-(const Tensor& tensor) {
  return negative(tensor);
}

/**
 * Performs element-wise logical-not on the elements of a tensor
 *
 * @param[in] tensor the tensor on which to perform logical not
 * @return a tensor with element-wise logical not of the input
 */
FL_API Tensor logicalNot(const Tensor& tensor);
inline Tensor operator!(const Tensor& tensor) {
  return logicalNot(tensor);
}

/**
 * Compute the element-wise exponential of a tensor
 *
 * @param[in] tensor the tensor to exponentiate
 * @return the exponentiated tensor
 */
FL_API Tensor exp(const Tensor& tensor);

/**
 * Compute the element-wise natural logarithm of a tensor
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
FL_API Tensor log(const Tensor& tensor);

/**
 * Returns the natural logarithm of one plus the input, element-wise.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
FL_API Tensor log1p(const Tensor& tensor);

/**
 * Returns the element-wise sine of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
FL_API Tensor sin(const Tensor& tensor);

/**
 * Returns the element-wise cosine of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
FL_API Tensor cos(const Tensor& tensor);

/**
 * Returns the element-wise non-negative square root of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
FL_API Tensor sqrt(const Tensor& tensor);

/**
 * Returns the element-wise hyperbolic tangent of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
FL_API Tensor tanh(const Tensor& tensor);

/**
 * Returns the element-wise floor of the input.
 *
 * @param[in] tensor the tensor on which to compute the floor
 * @return the resulting tensor
 */
FL_API Tensor floor(const Tensor& tensor);

/**
 * Returns the element-wise ceiling of the input.
 *
 * @param[in] tensor the tensor on which to compute the ceiling
 * @return the resulting tensor
 */
FL_API Tensor ceil(const Tensor& tensor);

/**
 * Returns the tensor with element-wise rounding to the nearest integer.
 *
 * @param[in] tensor the input tensor
 * @return the resulting tensor
 */
FL_API Tensor rint(const Tensor& tensor);

/**
 * Returns the element-wise absolute value of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
FL_API Tensor absolute(const Tensor& tensor);

// \copydoc absolute
inline Tensor abs(const Tensor& tensor) {
  return absolute(tensor);
}

/**
 * Returns the element-wise sigmoid the input:
 * \f[ out_i = \frac{1}{1 + \exp(-var_i)} \f]
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
FL_API Tensor sigmoid(const Tensor& tensor);

/**
 * Computes the element-wise error function the input: see
 * [here](https://en.wikipedia.org/wiki/Error_function) for details.
 *
 * @param[in] tensor the tensor on which to compute
 * @return ther resulting tensor
 */
FL_API Tensor erf(const Tensor& tensor);

/**
 * Flip a Tensor along a specified dimension.
 *
 * @param[in] tensor the tensor on which to compute
 * @param[in] dim the dimension along which to flip the tensor
 *
 * @return the resulting flipped tensor
 */
FL_API Tensor flip(const Tensor& tensor, const unsigned dim);

/**
 * Clip (limit) the values of a tensor. Given some interval of values, set
 * values outside of that interval to be the boundaries of the interval. All
 * values larger than the max become the max, and all values smaller than the
 * min become the min. Minimum and maximum values are determined element-wise in
 * the input `low` and `high` input tensors.
 *
 * \todo Require, enforce, and document broadcasting behavior in testing.
 *
 * @param[in] tensor the tensor to clip
 * @param[in] low a tensor containing minimum values used element-wise in
 * clipping
 * @param[in] high a tensor containing maximum values used element-wise in
 * clipping
 * @return a tensor with all values clipped between high and low
 */
FL_API Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high);

/**
 * Clip (limit) the values of a tensor. Given some interval of values, set
 * values outside of that interval to be the boundaries of the interval. All
 * values larger than the max become the max, and all values smaller than the
 * min become the min. Minimum values are determined element-wise in
 * the `low` input tensor and the upper bound scalar.
 *
 * @param[in] tensor the tensor to clip
 * @param[in] low a tensor containing minimum values used element-wise in
 * clipping
 * @param[in] high a scalar to use as the maximum value in clipping
 * @return a tensor with all values clipped between high and low
 */
FL_API Tensor clip(const Tensor& tensor, const Tensor& low, const double& high);

/**
 * Clip (limit) the values of a tensor. Given some interval of values, set
 * values outside of that interval to be the boundaries of the interval. All
 * values larger than the max become the max, and all values smaller than the
 * min become the min. Minimum values are given by the lower bound scalar and
 * element-wise in the `high` input tensor.
 *
 * @param[in] tensor the tensor to clip
 * @param[in] low a scalar to use as the minimum value in clipping
 * @param[in] high a tensor containing maximum values used element-wise in
 * clipping
 * @return a tensor with all values clipped between high and low
 */
FL_API Tensor clip(const Tensor& tensor, const double& low, const Tensor& high);

/**
 * Clip (limit) the values of a tensor. Given some interval of values, set
 * values outside of that interval to be the boundaries of the interval. All
 * values larger than the max become the max, and all values smaller than the
 * min become the min. Minimum values are determined by the passed scalars.
 *
 * @param[in] tensor the tensor to clip
 * @param[in] low a scalar to use as the minimum value in clipping
 * @param[in] high a scalar to use as the maximum value in clipping
 * @return a tensor with all values clipped between high and low
 */
FL_API Tensor clip(const Tensor& tensor, const double& low, const double& high);

/**
 * Rolls (or shifts) a tensor by a certain amount along a given axis, moving
 * elements that would be shifted out of bounds to the beginning of the axis in
 * a circular fashion.
 *
 * @param[in] tensor the tensor to roll shift
 * @param[in] shift the amount by which to shift
 * @param[in] axis the axis along which to perform the shift
 * @return a tensor with values shifted by the given amount in a circular
 * fashion
 */
FL_API Tensor roll(const Tensor& tensor, const int shift, const unsigned axis);

/**
 * Returns a boolean tensor which is true where the input tensor was NaN, and
 * false otherwise.
 *
 * @param[in] tensor the input tensor
 * @return a boolean tensor with true in positions that contained NaN in the
 * input tensor
 */
FL_API Tensor isnan(const Tensor& tensor);

/**
 * Returns a boolean tensor which is true where the input tensor was infinity,
 * and false otherwise.
 *
 * @param[in] tensor the input tensor
 * @return a boolean tensor with true in positions that contained Inf in the
 * input tensor
 */
FL_API Tensor isinf(const Tensor& tensor);

/**
 * Returns a tensor that contains -1 if an element is less than 0, 0 if an
 * element is 0, and 1 if an element is greater than zero. Returns NaN for NaN
 * values.
 *
 * @param[in] tensor the input tensor
 * @return a tensor containing element-wise sign values.
 */
FL_API Tensor sign(const Tensor& tensor);

/**
 * Returns an upper triangular version of the tensor.
 *
 * For tensors that have greater than two dimensions, this function outputs a
 * tensor with lower triangular submatrices along the last two dimensions of the
 * input tensor.
 *
 * @param[in] tensor the input tensor
 * @return a copy of the input tensor with elements above the diagonal zeroed
 * out
 */
FL_API Tensor tril(const Tensor& tensor);

/**
 * Returns an upper triangular version of the tensor.
 *
 * For tensors that have greater than two dimensions, this function outputs a
 * tensor with upper triangular submatrices along the last two dimensions of the
 * input tensor.
 *
 * @param[in] tensor the input tensor
 * @return a copy of the input tensor with elements below the diagonal zeroed
 * out
 */
FL_API Tensor triu(const Tensor& tensor);

/**
 * Conditionally return elements from one of two tensors based on a condition.
 *
 * @param[in] condition a tensor that, where true, uses values from x
 * positionally, else values from y. This tensor must be of type dtype::b8 else
 * an exception is thrown.
 * @param[in] x the tensor from which values are chosen for true values in
 * condition
 * @param[in] y the tensor from which values are chosen for false values in
 * condition
 *
 * @return the resulting tensor that contains elements of x where condition is
 * true and elements of y where condition is false.
 */
FL_API Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

/**
 * Conditionally return elements from a tensor or passed scalar based on a
 * condition.
 *
 * @param[in] condition a tensor that, where true, uses the scalar value x,
 * else positional values from y. This tensor must be of type dtype::b8 else
 * an exception is thrown.
 * @param[in] x the tensor from which values are chosen for true values in the
 * condition
 * @param[in] y the scalar returned for false values in the condition
 *
 * @return the resulting tensor that contains elements of x where condition is
 * true and the scalar value y where the condition is false.
 */
FL_API Tensor where(const Tensor& condition, const Tensor& x, const double& y);

/**
 * Conditionally return elements from a scalar or passed tensor based on a
 * condition.
 *
 * @param[in] condition a tensor that, where true, uses values from x
 * positionally, else the scalar value y. This tensor must be of type dtype::b8
 * else an exception is thrown.
 * @param[in] x the scalar returned for true values in the condition
 * @param[in] y the tensor from which values are chosen for false values in the
 * condition
 *
 * @return the resulting tensor that contains elements of x where condition is
 * true and the scalar value y where the condition is false.
 */
FL_API Tensor where(const Tensor& condition, const double& x, const Tensor& y);

/*!
 * Sorting mode for sorting-related functions.
 */
enum class SortMode { Descending = 0, Ascending = 1 };

/**
 * Get the top-k values and indices from a Tensor.
 *
 * @param[out] values the sorted tensor
 * @param[out] indices the indices corresponding to the sorted ordering
 * @param[in] input the input tensor to sort
 * @param[in] k the top number of elements to return
 * @param[in] axis the axis along which to sort.
 * @param[in] sortMode the ordering with which to sort. In descending mode, the
 * topk highest values are returned; else the topk lowest values are
 * returned. Defaults to descending.
 */
FL_API void topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode = SortMode::Descending);

/**
 * Sort the values of a tensor, and return the sorted tensor.
 *
 * @param[in] input the input Tensor
 * @param[in] axis the axis along which to sort
 * @param[in] sortMode the ordering with which to sort. Defaults to ascending
 */
FL_API Tensor sort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode = SortMode::Ascending);

/**
 * Sort the values of a tensor, and return the sorted tensor and sorted indices.
 *
 * @param[out] values the sorted tensor
 * @param[out] indices the indices corresponding to the sorted ordering
 * @param[in] input the input Tensor
 * @param[in] axis the axis along which to sort
 * @param[in] sortMode the ordering with which to sort. Defaults to ascending
 */
FL_API void sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode = SortMode::Ascending);

/**
 * Sort the values of a tensor and return the sorted indices.
 *
 * @param[in] input the input Tensor
 * @param[in] axis the axis along which to sort
 * @param[in] sortMode the ordering with which to sort. Defaults to ascending
 */
FL_API Tensor argsort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode = SortMode::Ascending);

/************************** Binary Operators ***************************/
// \cond DOXYGEN_DO_NOT_DOCUMENT
#define FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, TYPE)    \
  FL_API Tensor FUNC(TYPE lhs, const Tensor& rhs);        \
  FL_API Tensor FUNC(const Tensor& lhs, TYPE rhs);        \
  FL_API Tensor operator OP(TYPE lhs, const Tensor& rhs); \
  FL_API Tensor operator OP(const Tensor& lhs, TYPE rhs);

#define FL_BINARY_OP_LITERALS_DECL(OP, FUNC)                           \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const bool&);               \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const int&);                \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned&);           \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const char&);               \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned char&);      \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const long&);               \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned long&);      \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const long long&);          \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned long long&); \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const double&);             \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const float&);              \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const short&);              \
  FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned short&);

#define FL_BINARY_OP_DECL(OP, FUNC)                                \
  FL_API Tensor FUNC(const Tensor& lhs, const Tensor& rhs);        \
  FL_API Tensor operator OP(const Tensor& lhs, const Tensor& rhs); \
  FL_BINARY_OP_LITERALS_DECL(OP, FUNC);

FL_BINARY_OP_DECL(+, add);
FL_BINARY_OP_DECL(-, sub);
FL_BINARY_OP_DECL(*, mul);
FL_BINARY_OP_DECL(/, div);
FL_BINARY_OP_DECL(==, eq);
FL_BINARY_OP_DECL(!=, neq);
FL_BINARY_OP_DECL(<, lessThan);
FL_BINARY_OP_DECL(<=, lessThanEqual);
FL_BINARY_OP_DECL(>, greaterThan);
FL_BINARY_OP_DECL(>=, greaterThanEqual);
FL_BINARY_OP_DECL(||, logicalOr);
FL_BINARY_OP_DECL(&&, logicalAnd);
FL_BINARY_OP_DECL(%, mod);
FL_BINARY_OP_DECL(&, bitwiseAnd);
FL_BINARY_OP_DECL(|, bitwiseOr);
FL_BINARY_OP_DECL(^, bitwiseXor);
FL_BINARY_OP_DECL(<<, lShift);
FL_BINARY_OP_DECL(>>, rShift);

#undef FL_BINARY_OP_DECL
#undef FL_BINARY_OP_LITERALS_DECL
#undef FL_BINARY_OP_LITERAL_TYPE_DECL
// \endcond

/**
 * Returns the element-wise minimum of tensor elements.
 *
 * \todo Require, enforce, and document broadcasting behavior in testing.
 *
 * @param[in] lhs left hand side tensor for the minimum
 * @param[in] rhs right hand side tensor for the minimum
 * @return a tensor containing the minimum values in each tensor
 */
FL_API Tensor minimum(const Tensor& lhs, const Tensor& rhs);

/**
 * Returns the element-wise minimum of tensor elements with some scalar.
 *
 * @param[in] lhs the tensor
 * @param[in] rhs a scalar value
 * @return a tensor containing the minimum values element-wise with the tensor
 * and a scalar.
 */
FL_API Tensor minimum(const Tensor& lhs, const double& rhs);

/**
 * Returns the element-wise minimum of tensor elements with some scalar.
 *
 * @param[in] lhs a scalar value
 * @param[in] rhs the tensor
 * @return a tensor containing the minimum values element-wise with the tensor
 * and a scalar.
 */
FL_API Tensor minimum(const double& lhs, const Tensor& rhs);

/**
 * Returns the element-wise maximum of tensor elements.
 *
 * \todo Require, enforce, and document broadcasting behavior in testing.
 *
 * @param[in] lhs left hand side tensor for the minimum
 * @param[in] rhs right hand side tensor for the minimum
 * @return a tensor containing the maximum values in each tensor
 */
FL_API Tensor maximum(const Tensor& lhs, const Tensor& rhs);

/**
 * Returns the element-wise maximum of tensor elements with some scalar.
 *
 * @param[in] lhs the tensor
 * @param[in] rhs a scalar value
 * @return a tensor containing the maximum values element-wise with the tensor
 * and a scalar.
 */
FL_API Tensor maximum(const Tensor& lhs, const double& rhs);

/**
 * Returns the element-wise maximum of tensor elements with some scalar.
 *
 * @param[in] lhs a scalar value
 * @param[in] rhs the tensor
 * @return a tensor containing the maximum values element-wise with the tensor
 * and a scalar.
 */
FL_API Tensor maximum(const double& lhs, const Tensor& rhs);

/**
 * Returns the element-wise exponentiation of tensors; the left hand tensor is
 * exponentiated to the power of the right hand tensor, element-wise.
 *
 * @param[in] lhs the base tensor
 * @param[in] rhs the exponent tensor
 * @return a tensor containing the exponentiated values
 */
FL_API Tensor power(const Tensor& lhs, const Tensor& rhs);

/**
 * Returns the element-wise exponentiation of tensors raised to some scalar
 * power.
 *
 * @param[in] lhs the base tensor
 * @param[in] rhs a scalar exponent
 * @return a tensor containing the exponentiated values
 */
FL_API Tensor power(const Tensor& lhs, const double& rhs);

/**
 * Returns the element-wise exponentiation of a scalar raised element-wise to
 * values from a tensor.
 *
 * @param[in] lhs a scalar base
 * @param[in] rhs the tensor containing exponent values
 * @return a tensor containing the exponentiated values
 */
FL_API Tensor power(const double& lhs, const Tensor& rhs);

/******************************* BLAS ********************************/

/*!
 * Transformations to apply to Tensors (i.e. matrices) before applying certain
 * operations (i.e. matmul).
 */
enum class MatrixProperty { None = 0, Transpose = 1 };

/**
 * Perform matrix multiplication between two tensors.
 *
 * @param[in] lhs the Tensor on the left hand side
 * @param[in] rhs the Tensor on the right hand side
 * @param[in] lhsProp the `MatrixProperty` to apply to the tensor on the
 * left-hand side
 * @param[in] rhsProp the `MatrixProperty` to apply to the tensor on the
 * right-hand side
 *
 * @return an output tensor containing the matrix product.
 */
FL_API Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp = MatrixProperty::None,
    MatrixProperty rhsProp = MatrixProperty::None);

/************************** Reductions ***************************/

/**
 * Compute the minimum value along multiple axes. If axes is left empty,
 * computes the minumum along all axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce. If empty, computes along
 * all axes
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the max(es)
 */
FL_API Tensor amin(
    const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/**
 * Compute the maximum value along multiple axes. If axes is left empty,
 * computes the maximum along all axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce. If empty, computes along
 * all axes
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the max(es)
 */
FL_API Tensor amax(
    const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/**
 * Compute the maximum value along multiple axes for a tensor, returning both
 * the maximum values and the indices of the input tensor in which they appear.
 *
 * @param[out] values a Tensor into which to populate the max values from the
 * tensor along the specified axes
 * @param[out] indices a Tensor into which to populate the indices of the max
 * values from the tensor along the specified axes
 * @param[in] input the input tensor
 * @param[in] axis the axis along which to find minimum values
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 */
FL_API void min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims = false);

/**
 * Compute the maximum value along multiple axes for a tensor, returning both
 * the maximum values and the indices of the input tensor in which they appear.
 *
 * @param[out] values a Tensor into which to populate the max values from the
 * tensor along the specified axes
 * @param[out] indices a Tensor into which to populate the indices of the max
 * values from the tensor along the specified axes
 * @param[in] input the input tensor
 * @param[in] axis the axis along which to find maximum values
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 */
FL_API void max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims = false);

/**
 * Return the indices of the maximum values along an axis.
 *
 * @param[in] input the input tensor
 * @param[in] axis the axis along which to find maximum values
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the indices of the max values along each axis
 */
FL_API Tensor
argmax(const Tensor& input, const unsigned axis, const bool keepDims = false);

/**
 * Return the indices of the minimum values along an axis.
 *
 * @param[in] input the input tensor
 * @param[in] axis the axis along which to find minimum values
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the indices of the max values along each axis
 */
FL_API Tensor
argmin(const Tensor& input, const unsigned axis, const bool keepDims = false);

/**
 * Sum of tensor over given axes. If axes is left empty, computes the sum along
 * all axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce. If empty, computes along
 * all axes
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the sum(s)
 */
FL_API Tensor
sum(const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/**
 * Compute the cumulative sum (or the prefix sum, scan, or inclusive scan) of a
 * tensor along a given axis.
 *
 * @param[in] input the input tensor
 * @param[in] axis the axis along which to accumulate
 * @return a tensor of the same shape containing the accumulated sum
 */
FL_API Tensor cumsum(const Tensor& input, const unsigned axis);

/**
 * Mean of tensor over given axes. If axes is left empty, computes the mean
 * along all axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce. If empty, computes along
 * all axes
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the mean(s)
 */
FL_API Tensor mean(
    const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/**
 * Median of tensor over given axes. If axes is left empty, computes the median
 * along all axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce. If empty, computes along
 * all axes
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the median(s)
 */
FL_API Tensor median(
    const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/**
 * Variance of an tensor over given axes. If axes is left empty, computes the
 * variance along all axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce. If empty, computes along
 * all axes
 * @param[in] bias defaults false. Compute biased or unbiased variance
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the variance(s)
 */
FL_API Tensor
var(const Tensor& input,
    const std::vector<int>& axes = {},
    const bool bias = false,
    const bool keepDims = false);

/**
 * Standard deviation of an tensor over given axes. If axes is left empty,
 * computes the standard deviation along all axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce. If empty, computes along
 * all axes
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the standard deviation(s)
 */
FL_API Tensor
std(const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/**
 * Perform Lp-norm computation, reduced over specified dimensions. If axes is
 * left blank, computes the norm along all dimensions.
 *
 * @param[in] input tensor on which the Lp norm is going to be computed.
 * @param[in] p the p value of the Lp norm.
 * @param[in] axes dimensions over which the reduction is performed.
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the norm(s)
 */
FL_API Tensor norm(
    const Tensor& input,
    const std::vector<int>& axes = {},
    double p = 2,
    const bool keepDims = false);

/**
 * Counts the number of nonzero elements in a tensor.
 *
 * If k axes are passed, returns a tensor of size k with element-wise nonzero
 * counts along each axis.
 *
 * @param[in] input the tensor on which to operate.
 * @param[in] axes (optional) the axis along which to give nonzeros.
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a tensor containing the number of nonzero elements along each axis or
 * over the entire tensor.
 */
FL_API Tensor countNonzero(
    const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/**
 * Checks for any true values in a tensor along one or more axes; returns true
 * if any exist. If k axes are passed, returns a tensor of size k with
 * truthiness checks along each axis. If axes is left empty, computes the
 * variance along all axes.
 *
 * @param[in] input the input tensor
 * @param[in] axes the axes along which to check for truthy values. If empty,
 * computes along all axes
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a bool tensor containing axis-wise values denoting truthy values
 * along that axis in the input tensor.
 */
FL_API Tensor
any(const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/**
 * Checks if all values are true in a tensor along one or more axes; returns
 * true if all are true and false otherwise. If k axes are passed, returns a
 * tensor of size k with all-true checks along each axis. If axes is left empty,
 * computes the variance along all axes.
 *
 * @param[in] input the input tensor
 * @param[in] axes the axes along which to check. If empty, computes along
 * all axes
 * @param[in] keepDims defaults false. Keeps the dimensions being reduced over
 * as singleton dimensions rather than collapsing them
 * @return a bool tensor containing axis-wise values with true along
 * axes that contain only true values.
 */
FL_API Tensor
all(const Tensor& input,
    const std::vector<int>& axes = {},
    const bool keepDims = false);

/************************** Utilities ***************************/

/**
 * Write a string representation of a tensor to an output stream.
 */
FL_API std::ostream& operator<<(std::ostream& ostr, const Tensor& t);

/**
 * Print a string representation of a tensor to standard out.
 *
 * @param[in] tensor the tensor to print
 */
FL_API void print(const Tensor& tensor);

/**
 * Returns of two tensors are close. Checks:
 * - Tensor data types
 * - Tensor shapes
 * - Emptiness
 * - Absolute distance between elements
 *
 * @param[in] a lhs tensor
 * @param[in] b rhs tensor
 * @param[in] absTolerance the maximum-allowable distance between the
 * tensors
 */
FL_API bool allClose(
    const fl::Tensor& a,
    const fl::Tensor& b,
    const double absTolerance = 1e-5);

/**
 * @return if a Tensor contains any NaN or Inf values.
 */
FL_API bool isInvalidArray(const Tensor& tensor);

/**
 * Get a string representation of a tensor backend type.
 *
 * @param[in] type the tensor backend type.
 * @return a string representing the given tensor backend type.
 */
FL_API std::string tensorBackendTypeToString(const TensorBackendType type);

/**
 * Write a string representation of a tensor backend type to an output stream.
 *
 * @param[out] os the output stream.
 * @param[in] type the tensor backend type.
 * @return the output stream.
 */
FL_API std::ostream& operator<<(std::ostream& os, const TensorBackendType type);

/**
 * Convert a tensor from one type to another. Requires moving the input Tensor
 * - destroys the resulting tensor and creates another tensor of the desired
 * tensor type.
 *
 * @param[in] t the tensor to convert
 * @returns a tensor backed by the specified compile time type
 */
template <typename T>
Tensor to(Tensor&& t) {
  // Fast path -- types are the same
  if (T::tensorBackendType == t.backendType()) {
    return std::move(t);
  }

  if (t.isSparse()) {
    throw std::invalid_argument(
        "Tensor type conversion between sparse "
        "tensors not yet supported.");
  } else {
    // TODO: dynamically fix the memory location based on the type of
    // backend/where base memory is
    return Tensor(std::make_unique<T>(
        t.shape(), t.type(), t.device<void>(), MemoryLocation::Device));
  }
}

/** @} */

namespace detail {

bool areTensorTypesEqual(const Tensor& a, const Tensor& b);

template <typename... Args>
bool areTensorTypesEqual(
    const Tensor& a,
    const Tensor& b,
    const Args&... args) {
  return areTensorTypesEqual(a, b) && areTensorTypesEqual(a, args...);
}

} // namespace detail

/**
 * Checks if a variadic number of Tensors have the same type.
 */
#define FL_TENSOR_DTYPES_MATCH_CHECK(...)                                     \
  if (!detail::areTensorTypesEqual(__VA_ARGS__)) {                            \
    throw std::invalid_argument(                                              \
        std::string(__func__) + ": tensors are not all of the same types. "); \
  }

} // namespace fl
