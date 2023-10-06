/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/pkg/runtime/amp/DynamicScaler.h"

namespace fl {
namespace pkg {
namespace runtime {
/**
 * Get a certain checkpoint by `runidx`.
 */
std::string
getRunFile(const std::string& name, int runidx, const fs::path& runpath);

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
    std::shared_ptr<fl::pkg::runtime::DynamicScaler> dynamicScaler,
    std::shared_ptr<fl::Reducer> reducer);

/**
 * Returns the current date as a string
 */
std::string getCurrentDate();

/**
 * Returns the current time as a string
 */
std::string getCurrentTime();

} // namespace runtime
} // namespace pkg
} // namespace fl
