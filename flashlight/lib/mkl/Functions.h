/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

namespace fl {
namespace lib {
namespace mkl {

/**
 * Convolves the kernel on the input by delegating to MKL-VSL convolution.
 * Size of return value is kernel.size() + input.size() - 1
 */
std::vector<float> Correlate(
    const std::vector<float>& kernel,
    const std::vector<float>& input);

} // namespace mkl
} // namespace lib
} // namespace fl
