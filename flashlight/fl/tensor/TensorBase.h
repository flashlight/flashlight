/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

/**
 * Enum for various tensor backends.
 */
enum class TensorBackendType { ArrayFire };

// See TensorAdapter.h
class TensorAdapterBase;

// See TensorBackend.h
class TensorBackend;

// See Index.h
class Index;

/**
 * Location of memory or tensors.
 */
enum class Location { Host, Device };
// Alias to make it semantically clearer when referring to buffer location
using MemoryLocation = Location;

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
 * TODO:fl::Tensor {doc} more documentation. NOTE: this API will break
 * frequently and is not yet stable.
 */
class Tensor {
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
      void* ptr,
      MemoryLocation memoryLocation);

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
   * Create a tensor from a vector of values.
   *
   * @param[in] s the shape of the resulting tensor.
   * @param[in] vec values with which to populate the tensor.
   * @return a tensor with values and shape as given.
   */
  template <typename T>
  static Tensor fromVector(Shape s, std::vector<T> v) {
    return Tensor(s, fl::dtype_traits<T>::fl_type, v.data(), Location::Host);
  }

  template <typename T>
  static Tensor fromVector(std::vector<T> v) {
    return Tensor(
        {static_cast<long long>(v.size())},
        fl::dtype_traits<T>::fl_type,
        v.data(),
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
  static Tensor fromBuffer(Shape s, T* ptr, Location memoryLocation) {
    return Tensor(s, fl::dtype_traits<T>::fl_type, ptr, memoryLocation);
  }

  /**
   * Deep-copies the tensor, including underlying data.
   */
  Tensor copy() const;

  /**
   * Shallow-copies the tensor, returning a tensor that points to the same
   * underlying data.
   */
  Tensor shallowCopy() const;

  /**
   * Get the shape of a tensor.
   *
   * @return the shape of the tensor
   */
  const Shape& shape() const;

  /**
   * Get the number of elements in the tensor.
   *
   * @return the size of the tensor in elements.
   */
  size_t size() const;

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
   * Returns a tensor with elements cast as a particular type
   *
   * @param[in] the type to which to cast the tensor
   * @return a tensor with element-wise cast to the new type
   */
  Tensor astype(const dtype type);

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
   * @param[in] indices fl::Index instances to use
   * @return an indexed tensor
   */
  template <typename... Ts>
  Tensor operator()(const Ts&... args) const {
    std::vector<Index> indices{{args...}};
    return this->operator()(indices);
  }

  /**
   * Returns a representation of the tensor in 1 dimension.
   *
   * @return a 1D version of this tensor
   */
  Tensor flatten() const;

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
   * @return a scalar of the first element in the tensor.
   */
  template <typename T>
  T scalar() const;

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
   * Returns a pointer to the tensor's underlying data, but on the host. If the
   * tensor is located on a device, makes a copy of device memory and returns a
   * buffer on the host containing the relevant memory.
   *
   * @return the requested pointer on the host.
   */
  template <typename T>
  T* host() const;

  /**
   * Populates an existinb buffer with the tensor's underlying data, but on the
   * host. If the tensor is located on a device, makes a copy of device memory
   * and returns a buffer on the host containing the relevant memory.
   *
   * @return the requested pointer on the host.
   */
  template <typename T>
  void host(T* ptr) const;

  /**
   * Unlocks any device memory associated with the tensor that was acquired with
   * Tensor::device(), making it eligible to be freed.
   */
  void unlock() const;

  /**
   * Returns if the Tensor is contiguous in its memory-based representation.
   *
   * @return a bool denoting Tensor contiguousness
   */
  bool isContiguous() const;

  /******************** Assignment Operators ********************/
#define ASSIGN_OP(OP)                    \
  Tensor& OP(const Tensor& val);         \
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

  ASSIGN_OP(operator=);
  ASSIGN_OP(operator+=);
  ASSIGN_OP(operator-=);
  ASSIGN_OP(operator*=);
  ASSIGN_OP(operator/=);
#undef ASSIGN_OP
};

/******************** Tensor Creation Functions ********************/
/**
 * Creates a new tensor with a given shape and filled with a particular value.
 *
 * @param[in] dims the shape of the tensor to create
 * @param[in] val the value with which to fill the tensor
 * @param[in] type the type of the tensor to create. Defaults to a value based
 * on the value type
 * @return a tensor of the specified shape filled with the specified value
 */
template <typename T>
Tensor full(
    const Shape& dims,
    const T& val,
    const dtype type = dtype_traits<T>::ctype);

/**
 * Return a the identity tensor of a given size and type.
 *
 * @param[in] dim the size of the dimension of the matrix (dim x dim)
 * @param[in] type the type of the resulting matrix
 */
Tensor identity(const Dim dim, const dtype type = dtype::f32);

/************************ Shaping and Indexing *************************/

/**
 * Change a tensor's shape without changing its underlying data.
 *
 * @param[in] tensor the tensor to reshape
 * @param[in] shape the new shape for the tensor
 * @return the reshaped tensor
 */
Tensor reshape(const Tensor& tensor, const Shape& shape);

/**
 * Permute the axes of a tensor. If no arguments are given, reverses the axes of
 * a tensor.
 *
 * @param[in] tensor the tensor to transpose
 * @param[in] dims (optional) the permuted indices of the tensor the kth access
 * of the output tensor will correspond to dims[k] in the input tensor
 * @return the permuted tensor
 */
Tensor transpose(const Tensor& tensor, const Shape& dims = {});

/**
 * Repeat the contents of a tensor a given number of times along specified
 * dimensions.
 *
 * @param[in] tensor the tensor to tile
 * @param[in] shape the number of times, along each dimension, which to tile the
 * tensor
 * @return the tiled tensor
 */
Tensor tile(const Tensor& tensor, const Shape& shape);

/**
 * Join or concatenate tensors together along a particular axis.
 *
 * @param[in] tensors a vector of tensors to concatenate
 * @return a concatenated tensor
 */
Tensor concatenate(const std::vector<Tensor>& tensors, unsigned axis = 0);

/**
 * Join or concatenate tensors together along a particular axis.
 *
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
Tensor nonzero(const Tensor& tensor);

/************************** Unary Operators ***************************/
/**
 * Element-wise negation of a tensor.
 *
 * @param[in] tensor the input tensor to negate.
 * @return a tensor with elements negated.
 */
Tensor negative(const Tensor& tensor);
inline Tensor operator-(const Tensor& tensor) {
  return negative(tensor);
}

/**
 * Performs element-wise logical-not on the elements of a tensor
 *
 * @param[in] tensor the tensor on which to perform logical not
 * @return a tensor with element-wise logical not of the input
 */
Tensor logicalNot(const Tensor& tensor);
inline Tensor operator!(const Tensor& tensor) {
  return logicalNot(tensor);
}

/**
 * Compute the element-wise exponential of a tensor
 *
 * @param[in] tensor the tensor to exponentiate
 * @return the exponentiated tensor
 */
Tensor exp(const Tensor& tensor);

/**
 * Compute the element-wise natural logarithm of a tensor
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
Tensor log(const Tensor& tensor);

/**
 * Returns the natural logarithm of one plus the input, element-wise.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
Tensor log1p(const Tensor& tensor);

/**
 * Returns the element-wise sine of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
Tensor sin(const Tensor& tensor);

/**
 * Returns the element-wise cosine of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
Tensor cos(const Tensor& tensor);

/**
 * Returns the element-wise non-negative square root of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
Tensor sqrt(const Tensor& tensor);

/**
 * Returns the element-wise hyperbolic tangent of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
Tensor tanh(const Tensor& tensor);

/**
 * Returns the element-wise floor of the input.
 *
 * @param[in] tensor the tensor on which to compute the floor
 * @return the resulting tensor
 */
Tensor floor(const Tensor& tensor);

/**
 * Returns the element-wise ceiling of the input.
 *
 * @param[in] tensor the tensor on which to compute the ceiling
 * @return the resulting tensor
 */
Tensor ceil(const Tensor& tensor);

/**
 * Returns the element-wise absolute value of the input.
 *
 * @param[in] tensor the tensor on which to compute
 * @return the resulting tensor
 */
Tensor absolute(const Tensor& tensor);
inline Tensor abs(const Tensor& tensor) {
  return absolute(tensor);
}

/**
 * Clip (limit) the values of a tensor. Given some interval of values, set
 * values outside of that interval to be the boundaries of the interval. All
 * values larger than the max become the max, and all values smaller than the
 * min become the min.
 *
 * TODO: consider requiring broadcasting behavior/enforcing in a test
 *
 * @param[in] tensor the tensor to clip
 * @param[in] low a tensor containing
 * @param[in] high
 * @return a tensor with all values clipped between high and low
 */
Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high);
Tensor clip(const Tensor& tensor, const Tensor& low, const double& high);
Tensor clip(const Tensor& tensor, const double& low, const Tensor& high);
Tensor clip(const Tensor& tensor, const double& low, const double& high);

/**
 * Returns a boolean tensor which is true where the input tensor was NaN, and
 * false otherwise.
 *
 * @param[in] tensor the input tensor
 * @return a boolean array with true in positions that contained NaN in the
 * input tensor
 */
Tensor isnan(const Tensor& tensor);

/************************** Binary Operators ***************************/
#define FL_BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, TYPE) \
  Tensor FUNC(TYPE lhs, const Tensor& rhs);            \
  Tensor FUNC(const Tensor& lhs, TYPE rhs);            \
  Tensor operator OP(TYPE lhs, const Tensor& rhs);     \
  Tensor operator OP(const Tensor& lhs, TYPE rhs);

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

#define FL_BINARY_OP_DECL(OP, FUNC)                         \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs);        \
  Tensor operator OP(const Tensor& lhs, const Tensor& rhs); \
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
FL_BINARY_OP_DECL(|, bitwiseOr);
FL_BINARY_OP_DECL(^, bitwiseXor);
FL_BINARY_OP_DECL(<<, lShift);
FL_BINARY_OP_DECL(>>, rShift);

#undef FL_BINARY_OP_DECL
#undef FL_BINARY_OP_LITERALS_DECL
#undef FL_BINARY_OP_LITERAL_TYPE_DECL

/**
 * Returns the element-wise minimum of tensor elements.
 *
 * TODO: consider requiring broadcasting behavior/enforcing in a test
 *
 * @param[in] lhs left hand side tensor for the minimum
 * @param[in] rhs right hand side tensor for the minimum
 * @return a tensor containing the minimum values in each tensor
 */
Tensor minimum(const Tensor& lhs, const Tensor& rhs);
Tensor minimum(const Tensor& lhs, const double& rhs);
Tensor minimum(const double& lhs, const Tensor& rhs);

/**
 * Returns the element-wise maximum of tensor elements.
 *
 * TODO: consider requiring broadcasting behavior/enforcing in a test
 *
 * @param[in] lhs left hand side tensor for the minimum
 * @param[in] rhs right hand side tensor for the minimum
 * @return a tensor containing the maximum values in each tensor
 */
Tensor maximum(const Tensor& lhs, const Tensor& rhs);
Tensor maximum(const Tensor& lhs, const double& rhs);
Tensor maximum(const double& lhs, const Tensor& rhs);

/**
 * Returns the element-wise exponentiation of tensors; the left hand tensor is
 * exponentiated to the power of the right hand tensor, element-wise.
 *
 * @param[in] lhs the base tensor
 * @param[in] rhs the exponent tensor
 * @return a tensor containing the exponentiated values
 */
Tensor power(const Tensor& lhs, const Tensor& rhs);

/************************** Reductions ***************************/

/**
 * Compute the minimum value along multiple axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] dim the dimension along which to reduce.
 * @return a tensor containing the minimum values
 */
Tensor amin(const Tensor& input, const std::vector<int>& axes);

/**
 * Compute the minimum value across all axes.
 * TODO: consolidate with amin above.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the mim
 */
template <typename T>
T amin(const Tensor& input);

/**
 * Compute the maximum value along multiple axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] dim the dimension along which to reduce.
 * @return a tensor containing the max
 *
 */
Tensor amax(const Tensor& input, const std::vector<int>& axes);

/**
 * Compute the minimum value across all axes.
 * TODO: consolidate with amax above.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the mim
 */
template <typename T>
T amax(const Tensor& input);

/**
 * Sum of array over given axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce.
 * @return a tensor containing the sum across given axes
 */
Tensor sum(const Tensor& input, const std::vector<int>& axes);

/**
 * Sum of array over all axes.
 * TODO: consolidate with sum above.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the sum
 */
template <typename T>
T sum(const Tensor& input);

/**
 * Mean of array over given axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce.
 * @return a tensor containing the mean across given axes
 */
Tensor mean(const Tensor& input, const std::vector<int>& axes);

/**
 * Mean of array over all axes.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the mean
 */
template <typename T>
T mean(const Tensor& input);

/**
 * Variance of an array over given axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce.
 * @return a tensor containing the var across given axes
 */
Tensor
var(const Tensor& input, const std::vector<int>& axes, const bool bias = false);

/**
 * Variance of an array over all axes.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the var
 */
template <typename T>
T var(const Tensor& input, const bool bias = false);

/**
 * Standard deviation of an array over given axes.
 *
 * @param[in] input the input along which to operate
 * @param[in] axes the dimension along which to reduce.
 * @return a tensor containing the var across given axes
 */
Tensor std(const Tensor& input, const std::vector<int>& axes);

/**
 * norm of array over all axes.
 *
 * @param[in] input the input along which to operate
 * @return a double containing the norm
 */
double norm(const Tensor& input);

/**
 * Counts the number of nonzero elements in a tensor.
 *
 * If k axes are passed, returns a tensor of size k with element-wise nonzero
 * counts along each axis.
 *
 * @param[in] input the tensor on which to operate.
 * @param[in] dims (optional) the axis along which to give nonzeros.
 *
 * @return a tensor containing the number of nonzero elements along each axis or
 * over the entire tensor.
 */
Tensor countNonzero(const Tensor& input, const std::vector<int>& axes = {});

/************************** Utilities ***************************/

/**
 * Print a string representation of a tensor to standard out.
 *
 * @param[in] tensor the tensor to print
 */
void print(const Tensor& tensor);

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
bool allClose(
    const fl::Tensor& a,
    const fl::Tensor& b,
    const double absTolerance = 1e-5);

} // namespace fl
