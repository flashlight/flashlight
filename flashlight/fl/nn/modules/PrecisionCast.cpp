/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/PrecisionCast.h"

#include "flashlight/fl/common/Utils.h"

namespace fl {

PrecisionCast::PrecisionCast(fl::dtype targetType) : targetType_(targetType) {}

std::vector<Variable> PrecisionCast::forward(
    const std::vector<Variable>& inputs) {
  std::vector<Variable> outputs;
  for (const auto& input : inputs) {
    auto output = input.astype(targetType_);
    outputs.push_back(output);
  }
  return outputs;
}

Variable PrecisionCast::forward(const Variable& input) {
  return forward(std::vector<Variable>{input}).front();
}

Variable PrecisionCast::operator()(const Variable& input) {
  return this->forward(input);
}

std::unique_ptr<Module> PrecisionCast::clone() const {
  return std::make_unique<PrecisionCast>(*this);
}

std::string PrecisionCast::prettyString() const {
  std::ostringstream ss;
  ss << "PrecisionCast";
  ss << " * -> " << targetType_;
  return ss.str();
}

} // namespace fl
