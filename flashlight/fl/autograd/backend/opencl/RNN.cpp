/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"

namespace fl {

// TODO, implement RNN, also check to ensure there will appropriate checks to
// guard the use of half precision in case CPU implementation doesn't support
// it.
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
  throw std::runtime_error("rnn not yet implemented on opencl");
}

} // namespace fl
