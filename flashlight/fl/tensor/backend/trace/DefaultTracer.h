/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/trace/TracerBase.h"

namespace fl {

/**
 * A simple tracer that writes operations to an output stream using a
 * function call-like format. Provides an interface for argument types to be
 * written as strings - can be overridden to specify custom formats for writing
 * out arbitrary data.
 */
class DefaultTracer : public TracerBase {
 public:
  DefaultTracer(std::unique_ptr<std::ostream> stream);
  virtual ~DefaultTracer() = default;

  std::string traceArgumentList(ArgumentList args) override;
  void trace(TraceData data) override;
  void trace(
      const std::string& funcName,
      ArgumentList args,
      ArgumentList inputs,
      ArgumentList outputs) override;

  virtual std::string toTraceString(bool b);
  virtual std::string toTraceString(const Shape& shape);
  virtual std::string toTraceString(const dtype& type);
  virtual std::string toTraceString(const Tensor& tensor);
  virtual std::string toTraceString(
      std::reference_wrapper<const Tensor> tensor);
  virtual std::string toTraceString(const range& range);
  virtual std::string toTraceString(const Dim& range);
  virtual std::string toTraceString(const Index& index);
  std::string toTraceString(const std::vector<Index>& indices);
  virtual std::string toTraceString(
      std::reference_wrapper<const std::vector<Index>> indices);
  virtual std::string toTraceString(const std::vector<int>& vec);
  virtual std::string toTraceString(
      const std::vector<std::pair<int, int>>& vec);
  virtual std::string toTraceString(const std::vector<Tensor>& tensors);
  virtual std::string toTraceString(
      std::reference_wrapper<const std::vector<Tensor>> tensors);
  // TODO(jacobkahn): move to enum toString methods and call those directly
  virtual std::string toTraceString(const SortMode& sortMode);
  virtual std::string toTraceString(const PadType& padType);
  virtual std::string toTraceString(const MatrixProperty& matrixProperty);
  virtual std::string toTraceString(const Location& location);
  virtual std::string toTraceString(const StorageType& storageType);

  // FIXME: this is completely unnecessary. Someone is decaying somewhere, I
  // can't figure out who it is, and they are evil. Whatever.
  template <typename T>
  std::string toTraceString(std::reference_wrapper<const T> type) {
    std::stringstream ss;
    ss << toTraceString(type.get());
    return ss.str();
  }

  template <typename T>
  std::string toTraceString(const T& type) {
    std::stringstream ss;
    ss << type;
    return ss.str();
  }
};

} // namespace fl
