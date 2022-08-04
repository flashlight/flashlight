/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/trace/TensorTracer.h"

#include <iomanip>
#include <stdexcept>

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

TensorTracer::TensorTracer(std::unique_ptr<std::ostream> stream)
    : DefaultTracer(std::move(stream)) {}

std::string TensorTracer::toTraceString(const Tensor& tensor) {
  std::stringstream ss;
  ss << "{" << std::quoted("tensor") << ": {" << std::quoted("shape") << ": "
     << DefaultTracer::toTraceString(tensor.shape()) << ", "
     << std::quoted("type") << ": " << std::quoted(dtypeToString(tensor.type()))
     << ", " << std::quoted("device") << ": "
     << tensor.stream().device().nativeId() << ", " << std::quoted("backend")
     << ": " << std::quoted(tensorBackendTypeToString(tensor.backendType()))
     << ", "
     // TODO: we know this is a traceable Tensor, so get an ID [if we add ID]
     //  << std::quoted("id") << ": " << tensor.id() << ", "
     << std::quoted("memlocation") << ": \"" << tensor.device<void>() << "\"}}";
  tensor.unlock();
  return ss.str();
}

} // namespace fl
