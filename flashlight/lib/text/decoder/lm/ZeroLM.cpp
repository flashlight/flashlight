/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/text/decoder/lm/ZeroLM.h"

#include <stdexcept>

namespace fl {
namespace lib {
namespace text {

LMStatePtr ZeroLM::start(bool /* unused */) {
  return std::make_shared<LMState>();
}

std::pair<LMStatePtr, float> ZeroLM::score(
    const LMStatePtr& state /* unused */,
    const int usrTokenIdx) {
  return std::make_pair(state->child<LMState>(usrTokenIdx), 0.0);
}

std::pair<LMStatePtr, float> ZeroLM::finish(const LMStatePtr& state) {
  return std::make_pair(state, 0.0);
}
} // namespace text
} // namespace lib
} // namespace fl
