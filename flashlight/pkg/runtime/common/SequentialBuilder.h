/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/contrib/modules/modules.h"

namespace fl {

class Tensor;

namespace pkg {
namespace runtime {

/**
 * Build a sequential module by parsing a file that
 * defines the model architecture.
 */
std::shared_ptr<fl::Sequential> buildSequentialModule(
    const fs::path& archfile,
    int64_t nFeatures,
    int64_t nClasses);

/**
 * Utility function for to run forward with pad masking
 * casting of modules happens to use pad masking for trasnfromer layers
 * properly. It assumes that model is constructed with
 * buildSequentialModule. Caveat: it is not supporting resnet block
 * with a transformer block in it!
 * TODO remove with landing plugin arch instead of arch files
 */
fl::Variable forwardSequentialModuleWithPadMask(
    const fl::Variable& input,
    std::shared_ptr<fl::Module> ntwrk,
    const Tensor& inputSizes);

} // namespace runtime
} // namespace pkg
} // namespace fl
