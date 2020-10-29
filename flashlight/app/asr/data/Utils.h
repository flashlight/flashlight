/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"

namespace fl {
namespace app {
namespace asr {

std::vector<std::string> wrd2Target(
    const std::string& word,
    const lib::text::LexiconMap& lexicon,
    const lib::text::Dictionary& dict,
    bool fallback2Ltr = false,
    bool skipUnk = false);

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const lib::text::LexiconMap& lexicon,
    const lib::text::Dictionary& dict,
    bool fallback2Ltr = false,
    bool skipUnk = false);

std::vector<std::string> wrd2Target(
    const std::string& word,
    const lib::text::LexiconMap& lexicon,
    const lib::text::Dictionary& dict,
    float targetSamplePct = 0,
    bool fallback2Ltr = false,
    bool skipUnk = false);

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const lib::text::LexiconMap& lexicon,
    const lib::text::Dictionary& dict,
    const std::string& wordSeparator = "",
    float targetSamplePct = 0,
    bool fallback2Ltr = false,
    bool skipUnk = false);
} // namespace asr
} // namespace app
} // namespace fl
