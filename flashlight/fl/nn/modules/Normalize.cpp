/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Normalize.h"
#include "flashlight/fl/autograd/Functions.h"

namespace fl {

Normalize::Normalize(
    const std::vector<int>& axes,
    double p /* = 2 */,
    double eps /* = 1e-12 */,
    double value /* = 1 */)
    : axes_(axes), p_(p), eps_(eps), value_(value) {}

Variable Normalize::forward(const Variable& input) {
  return value_ * normalize(input, axes_, p_, eps_);
}

std::unique_ptr<Module> Normalize::clone() const {
  return std::make_unique<Normalize>(*this);
}

std::string Normalize::prettyString() const {
  std::ostringstream ss;
  ss << "Normalize";
  ss << " ( axis : { ";
  for (auto d : axes_) {
    ss << d << " ";
  }
  ss << "} , p : " << p_;
  ss << ", eps : " << eps_;
  ss << ", value : " << value_;
  ss << " )";
  return ss.str();
}

} // namespace fl
