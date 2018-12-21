/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/autograd/Functions.h>
#include <flashlight/autograd/Variable.h>
#include <flashlight/common/Exception.h>

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
  AFML_THROW_ERR("CPU RNN is not yet supported.", AF_ERR_NOT_SUPPORTED);
}

} // namespace fl
