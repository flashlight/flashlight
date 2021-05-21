/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Generic utilities which should not depend on ArrayFire / flashlight.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "flashlight/pkg/speech/decoder/TranscriptionUtils.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

namespace fl {
namespace app {
namespace asr {

/* A series of vector to vector mapping operations */

std::shared_ptr<fl::lib::text::Trie> buildTrie(
    const std::string& decoderType,
    bool useLexicon,
    std::shared_ptr<fl::lib::text::LM> lm,
    const std::string& smearing,
    const fl::lib::text::Dictionary& tokenDict,
    const fl::lib::text::LexiconMap& lexicon,
    const fl::lib::text::Dictionary& wordDict,
    const int wordSeparatorIdx,
    const int repLabel);

} // namespace asr
} // namespace app
} // namespace fl
