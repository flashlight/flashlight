/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

/**
 * A trait to turn an arbitrary Tensor type into a JitTensorBase, i.e.,
 * `JitTensor<T>` is a JitTensorBase that wraps around the tensor backend `T`.
 */
template <typename T>
class JitTensor : public JitTensorBase {
 protected:
  Tensor fromNode(Node* node) const override {
    return toTensor<JitTensor>(node);
  }

  TensorBackend& wrappedBackend() const override {
    static TensorBackend& wrappedBackend = toTensor<T>().backend();
    return wrappedBackend;
  }

  Evaluator& evaluator() const override {
    static Evaluator evaluator(wrappedBackend());
    return evaluator;
  }

 public:
  // 1 static instance per jitted T.
  // NOTE that it's safe even for multiple translation units:
  // https://stackoverflow.com/questions/19366615/static-member-variable-in-class-template
  JitBackend& backend() const override {
    auto creator = [](Node* node) { return toTensor<JitTensor>(node); };
    static JitBackend backend(wrappedBackend(), creator);
    return backend;
  }

  explicit JitTensor(Node* node) : JitTensorBase(node) {}

  // TODO SPoC for these defaults (also in OneDNN backend)
  JitTensor() : JitTensor({0}, fl::dtype::f32, nullptr, Location::Host) {}

  /**
   * Construct a JitTensorBase using some data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] type the type of the new tensor
   * @param[in] memoryLocation the location of the buffer
   */
  JitTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation)
      : JitTensorBase(
            ValueNode::create(toTensor<T>(shape, type, ptr, memoryLocation))) {}

  // Constructor for a sparse JitTensorBase
  JitTensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType)
      : JitTensorBase(ValueNode::create(
            toTensor<T>(nRows, nCols, values, rowIdx, colIdx, storageType))) {}

  std::unique_ptr<TensorAdapterBase> clone() const override {
    // TODO copy node
    throw std::runtime_error("[JitTensor] clone not supported");
  }
};

} // namespace fl
