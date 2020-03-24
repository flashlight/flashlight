/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/distributed/distributed.h"
#include "flashlight/nn/modules/Module.h"

namespace fl {

/**
 * Registers a module for allreduce synchronization with a gradient hook on it
 * parameter Variables.
 *
 * @param[in] module a module whose parameter gradients will be synchronized
 * @param[in] a ``Reducer`` instance to which gradients will be immediately
 * added when available
 */
void distributeModuleGrads(
    std::shared_ptr<const Module> module,
    std::shared_ptr<Reducer> reducer =
        std::make_shared<InlineReducer>(1.0 / getWorldSize()));

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
