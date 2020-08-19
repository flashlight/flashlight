/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include "libraries/language/dictionary/Dictionary.h"
#include "libraries/language/dictionary/Utils.h"

namespace fl {
namespace task {
namespace asr {

std::vector<std::string> wrd2Target(
    const std::string& word,
    const lib::LexiconMap& lexicon,
    const lib::Dictionary& dict,
    bool fallback2Ltr = false,
    bool skipUnk = false);

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const lib::LexiconMap& lexicon,
    const lib::Dictionary& dict,
    bool fallback2Ltr = false,
    bool skipUnk = false);
} // namespace asr
} // namespace task
} // namespace fl
