/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

#include <kenlm/lm/model.hh>

namespace fl {
namespace lib {
namespace text {

/**
 * KenLMState is a state object from KenLM, which  contains context length,
 * indicies and compare functions
 * https://github.com/kpu/kenlm/blob/master/lm/state.hh.
 */

struct KenLMState : LMState {
  lm::ngram::State ken_;
  lm::ngram::State* ken() {
    return &ken_;
  }
};

/**
 * KenLM extends LM by using the toolkit https://kheafield.com/code/kenlm/.
 */
class KenLM : public LM {
 public:
  KenLM(const std::string& path, const Dictionary& usrTknDict);

  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr& state,
      const int usrTokenIdx) override;

  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override;

 private:
  std::shared_ptr<lm::base::Model> model_;
  const lm::base::Vocabulary* vocab_;
};

using KenLMPtr = std::shared_ptr<KenLM>;
} // namespace text
} // namespace lib
} // namespace fl
