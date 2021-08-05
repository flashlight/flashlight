/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/lib/text/decoder/lm/LM.h"

namespace fl {
namespace lib {
namespace text {

/**
 * ZeroLM is a dummy language model class, which mimics the behavious of a
 * uni-gram language model but always returns 0 as score.
 */
class ZeroLM : public LM {
 public:
  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr& state,
      const int usrTokenIdx) override;

  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override;
};
} // namespace text
} // namespace lib
} // namespace fl
