/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/nn/modules/Module.h"

namespace fl {

/**
 * Registers a module for allreduce synchronization with a gradient hook on it
 * parameter Variables.
 *
 * @param module a module whose parameter gradients will be synchronized
 * @param scale scale gradients after allreduce by this factor
 */
void distributeModuleGrads(
    std::shared_ptr<const Module> module,
    double scale = 1.0);

/**
 * Traverses the network and averages its parameters with allreduce.
 *
 * @param module a module whose parameters will be synchronized
 */
void allReduceParameters(std::shared_ptr<const Module> module);

/**
 * Traverses the network and synchronizes the gradients of its parameters with
 * allreduce.
 *
 * @param module a module whose parameter gradients will be synchronized
 * @param scale scale gradients after allreduce by this factor
 */
void allReduceGradients(
    std::shared_ptr<const Module> module,
    double scale = 1.0);

} // namespace fl
