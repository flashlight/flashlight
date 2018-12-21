/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/autograd/Functions.h>

#include <flashlight/autograd/Variable.h>
#include <flashlight/common/Exception.h>

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
  AFML_THROW_ERR("CPU BatchNorm is not implemented yet.", AF_ERR_NOT_SUPPORTED);
}

} // namespace fl
