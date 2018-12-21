/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Transform.h"

#include <flashlight/autograd/Variable.h>

namespace fl {

Transform::Transform(
    const std::function<Variable(const Variable&)>& func,
    const std::string& name /* = "" */)
    : func_(func), name_(name) {}

Variable Transform::forward(const Variable& input) {
  return func_(input);
}

std::string Transform::prettyString() const {
  std::ostringstream ss;
  ss << "Transform ('" << name_ << "')";
  return ss.str();
}

} // namespace fl
