/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/flashlight/autograd/Functions.h"
#include "flashlight/flashlight/autograd/Variable.h"

namespace fl {

// TODO, implement RNN
std::tuple<Variable, Variable, Variable> rnn(
    const Variable& /* input */,
    const Variable& /* hidden_state */,
    const Variable& /* cell_state */,
    const Variable& /* weights */,
    int /* hidden_size */,
    int /* num_layers */,
    RnnMode /* mode */,
    bool /* bidirectional */,
    float /* dropout */) {
  throw std::runtime_error("rnn not yet implemented on CPU");
}

} // namespace fl
