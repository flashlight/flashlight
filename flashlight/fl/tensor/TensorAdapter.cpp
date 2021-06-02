/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorAdapter.h"

#include <memory>
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

#if FL_USE_ARRAYFIRE
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#endif

namespace fl {
namespace detail {

/*
 * Resolve the default tensor backend based on compile-time dependencies.
 *
 * For now, ArrayFire is required. If not available, throw.
 */
std::unique_ptr<TensorAdapterBase> getDefaultAdapter() {
#if FL_USE_ARRAYFIRE
  return std::make_unique<ArrayFireTensor>();
#else
  throw std::runtime_error(
      "Cannot construct tensor: Flashlight built "
      "without an available tensor backend.");
#endif
}

} // namespace detail
} // namespace fl
