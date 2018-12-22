/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include <flashlight/autograd/Functions.h>
#include <flashlight/autograd/Variable.h>

namespace fl {

Variable batchnorm(
    const Variable& /* input */,
    const Variable& /* weight */,
    const Variable& /* bias */,
    Variable& /* running_mean */,
    Variable& /* running_var */,
    const std::vector<int>& /* axes */,
    bool /* train */,
    double /* momentum */,
    double /* epsilon */) {
  throw std::runtime_error("batchnorm not yet implemented on CPU");
}

} // namespace fl
