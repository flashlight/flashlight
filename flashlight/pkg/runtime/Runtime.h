/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/runtime/amp/DynamicScaler.h"
#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
/**
 * Get a certain checkpoint by `runidx`.
 */
std::string
getRunFile(const std::string& name, int runidx, const std::string& runpath);

/**
 * Serialize gflags into a buffer.
 */
std::string serializeGflags(const std::string& separator = "\n");

/**
 * Properly scale the loss for back-propogation.
 * Return false when NAN or INF occurs in gradients, true otherwise.
 *
 * @param[in] loss - the loss to back propogate from.
 * @param[in] params - the whole set of learnable parameters.
 * @param[in] dynamicScaler - dynamic scaler to scale the loss and unscale the
 * gradients.
 * @param[in] reducer - to synchronize gradients in back-propogation.
 */
bool backwardWithScaling(
    const fl::Variable& loss,
    std::vector<fl::Variable>& params,
    std::shared_ptr<fl::ext::DynamicScaler> dynamicScaler,
    std::shared_ptr<fl::Reducer> reducer);

} // namespace app
} // namespace fl
