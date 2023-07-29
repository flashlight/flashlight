/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

std::shared_ptr<Module> loadOnnxModule();

} // namespace fl
