/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/trace/TracerBackendBase.h"

namespace fl {

class TracerTensorBase;

/**
 * Base implementation for tracer tensors.
 */
class TracerTensorBase : public TensorAdapterBase {
  /*
   * Return the TensorBackend implementation whose op calls are being traced.
   */
  virtual TensorBackend& tracedBackend() const = 0;

  /*
   * The TracerBackendBase instance corresponding to this tensor.
   */
  virtual TracerBackendBase& backend() const = 0;

  // Private -- throws.
  TracerTensorBase(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation);

  // Private -- throws.
  TracerTensorBase(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

 protected:
  // The tensor that this TracerTensor wraps.
  Tensor tracedTensor_;

 public:
  TracerTensorBase(Tensor tensor) : tracedTensor_(std::move(tensor)) {}

  Tensor& tensor() {
    return tracedTensor_;
  }

  ~TracerTensorBase() override = default;
  std::unique_ptr<TensorAdapterBase> clone() const override;
  TensorBackendType backendType() const override;

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
#define ASSIGN_OP_TYPE(OP, TYPE) virtual void OP(const TYPE& val) override;
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
