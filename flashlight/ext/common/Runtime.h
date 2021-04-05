/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fl {
namespace ext {
/**
 * Get a certain checkpoint by `runidx`.
 */
std::string
getRunFile(const std::string& name, int runidx, const std::string& runpath);

/**
 * Serialize gflags into a buffer.
 */
std::string serializeGflags(const std::string& separator = "\n");

} // namespace ext
} // namespace fl
