/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Padding.h"

#include "flashlight/fl/autograd/Functions.h"

namespace fl {

Padding::Padding(std::pair<int, int> pad0, double val)
    : m_pad({pad0}), m_val(val) {}

Padding::Padding(std::pair<int, int> pad0, std::pair<int, int> pad1, double val)
    : m_pad({pad0, pad1}), m_val(val) {}

Padding::Padding(
    std::pair<int, int> pad0,
    std::pair<int, int> pad1,
    std::pair<int, int> pad2,
    double val)
    : m_pad({pad0, pad1, pad2}), m_val(val) {}

Padding::Padding(
    std::pair<int, int> pad0,
    std::pair<int, int> pad1,
    std::pair<int, int> pad2,
    std::pair<int, int> pad3,
    double val)
    : m_pad({pad0, pad1, pad2, pad3}), m_val(val) {}

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
