/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <ostream>

#include "flashlight/fl/tensor/backend/trace/DefaultTracer.h"

namespace fl {

/**
 * A tracer that traces special Tensor metadata (tensor device, address in
 * memory for identifying tensors over time)
 */
class TensorTracer : public DefaultTracer {
 public:
  explicit TensorTracer(std::unique_ptr<std::ostream> stream);

  // List richer tensor metadata in the generated trace
  std::string toTraceString(const Tensor& tensor) override;
};

} // namespace fl
