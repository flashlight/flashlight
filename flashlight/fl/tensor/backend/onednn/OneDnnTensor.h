/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorAdapter.h"

#include <dnnl.hpp>

namespace fl {

/**
 * Tensor adapter implemented using the OneDNN library. Maps operations
 * expressed in Flashlight Tensors to OneDNN.
 */
class OneDnnTensor : public TensorAdapterBase {
  /**
   * We store the memory as (a). row-major with (b). dimensions reversed because
   * 1. dnnl::memory::desc::reshape only works on row-major layout.
   * 2. Flashlight requires column-major layout.
   *
   * NOTE (b) allows the index conversion to be a simple reversal, because the
   * logical and internal representations are isomorphic via reversing all the
   * axes, i.e., transpose({N-1, ..., 0}).
   *
   * Example:
   *   For shape (3, 2) and data [1, ..., 6], the logical Flashlight Tensor is
   *       [[1, 4],
   *        [2, 5],
   *        [3, 6]]
   *
   *   whereas the underlying OneDnnTensor representation is:
   *     fl::Shape   = (3, 2)      # same as logical representation
   *     dnnl::memory
   *         dims    = (2, 3)      # reversed
   *         data    = [1, ..., 6] # same as logical representation
   *         strides = (3, 1)      # reversed
   *
   *   Visually, the dnnl::memory internally represents such a tensor:
   *       [[1, 2, 3],
   *        [4, 5, 6]]
   */
  dnnl::memory memory_;
  Shape shape_;

 public:
  /**
   * Construct an OneDNNTensor with given shape and memory.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] memory the memory handle containing underlying tensor data
   */
  OneDnnTensor(const Shape& shape, dnnl::memory&& memory);

  /**
   * Construct an empty OneDNNTensor.
   */
  OneDnnTensor();

  /**
   * Construct a OneDNNTensor using some data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] type the type of the new tensor
   * @param[in] memoryLocation the location of the buffer
   */
  OneDnnTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation);

  // Constructor for a sparse OneDNNTensor; currently not supported.
  OneDnnTensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  ~OneDnnTensor() override = default;
  std::unique_ptr<TensorAdapterBase> clone() const override;
  TensorBackendType backendType() const override;
  TensorBackend& backend() const override;
  Tensor copy() override;
  Tensor shallowCopy() override;
  const Shape& shape() override;
  dtype type() override;
  bool isSparse() override;
  Location location() override;
  void scalar(void* out) override;
  void device(void** out) override;
  void host(void* out) override;
  void unlock() override;
  bool isLocked() override;
  bool isContiguous() override;
  Shape strides() override;
  const Stream& stream() const override;
  Tensor astype(const dtype type) override;
  Tensor index(const std::vector<Index>& indices) override;
  Tensor flatten() const override;
  Tensor flat(const Index& idx) const override;
  Tensor asContiguousTensor() override;
  void setContext(void* context) override;
  void* getContext() override;
  std::string toString() override;
  std::ostream& operator<<(std::ostream& ostr) override;

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

} // namespace fl
