/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gflags/gflags.h>

namespace fl {
namespace app {
namespace lm {

std::string serializeGflags(const std::string& separator = "\n");

} // namespace lm
} // namespace app
} // namespace fl
