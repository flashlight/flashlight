/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/algorithm.h>
#include <af/array.h>
#include <af/statistics.h>

#include <variant>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorAdapter.h"

namespace fl {

/**
 * Tensor adapter for the ArrayFire tensor library. Maps operations expressed in
 * Flashlight Tensors to ArrayFire
 */
class ArrayFireTensor : public TensorAdapterBase {
  // The internal ArrayFire handle. This can be an af::array or an
  // af::array::array_proxy as in the case of a view (lvalues only)
  std::variant<af::array, af::array::array_proxy> handle_;

  /**
   * Internal getter for a handle. If the handle is an array_proxy, promote to
   * an array, condense dimensions as needed, replace the handle variant, and
   * return a reference.
   */
  af::array& handle();

  /*
   * A Flashlight shape that mirrors ArrayFire dims.
   *
   * NOTE: this shape is only updated on calls to ArrayFireTensor::shape() so
   * as to satisfy API requirements. af::array::dims() should be used for
   * internal computation where shape/dimensions are needed.
   */
  Shape shape_;

  /**
   * Constructs an ArrayFireTensor from an af::array::array_proxy.
   *
   * This constructor is for internal use only. In order to properly construct
   * views/proxies of arrays when they're lvalues, we need to sometimes create
   * ArrayFireTensors which hold proxies and not arrays.
   *
   * Whenever these ArrayFireTensors are mutated, ArrayFireTensor::handle() is
   * called, which upcasts the array_proxy to a full af::array on which
   * operations can be performed.
   *
   * @param[in] arrayProxy construct a tensor from an ArrayFire array_proxy
   * rvalue reference.
   */
  explicit ArrayFireTensor(af::array::array_proxy&& arrayProxy);

 public:
  /**
   * Constructs an ArrayFireTensor.
   *
   * Since af::arrays are refcounted, an instance of this class
   * can only be created using arrays that are moved therein.
   *
   * Tensor operations occurring directly on this tensor's underlying
   * af::array should not copy the array else take a performance penalty (via
   * an internal copy if refcount is > 1 in some cases).
   *
   * @param[in] array construct a tensor from an ArrayFire array rvalue
   * reference.
   */
  explicit ArrayFireTensor(af::array&& array);

  /**
   * Default initialization - empty ArrayFire array and empty shape.
   */
  ArrayFireTensor();

  /**
   * Gets an ArrayFire Array from this impl.
   */
  const af::array& getHandle() const;

  /**
   * Gets an ArrayFire Array from this impl. If the underlying handle is an
   * array_proxy, may promote it to an af::array.
   */
  af::array& getHandle();

  ~ArrayFireTensor() override = default;
  std::unique_ptr<TensorAdapterBase> clone() const override;
  TensorBackendType backendType() const override;
  TensorBackend& backend() const override;
  const Shape& shape() override;
  dtype type() override;
  Tensor astype(const dtype type) override;
  Tensor index(const std::vector<Index>& indices) override;
  Tensor flatten() const override;

  /******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(OP, TYPE) void OP(const TYPE& val) override;

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

/**
 * Gets an af::array from a Tensor. If the Tensor is not ArrayFire-backed,
 * throws an exception
 *
 * @param[in] tensor the input tensor
 * @return the array underying the Tensor
 */
const af::array& toArray(const Tensor& tensor);
af::array& toArray(Tensor& tensor);

} // namespace fl
