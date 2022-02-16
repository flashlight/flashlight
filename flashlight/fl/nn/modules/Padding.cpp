/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Padding.h"

#include "flashlight/fl/autograd/Functions.h"

namespace fl {

Padding::Padding(std::vector<std::pair<int, int>> padding, double val)
    : m_pad(std::move(padding)), m_val(val) {}

Variable Padding::forward(const Variable& input) {
  return padding(input, m_pad, m_val);
}

std::string Padding::prettyString() const {
  std::ostringstream ss;
  ss << "Padding (" << m_val << ", { ";
  for (auto p : m_pad) {
    ss << "(" << p.first << ", " << p.second << "), ";
  }
  ss << "})";
  return ss.str();
}

} // namespace fl
