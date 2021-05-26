/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include "flashlight/pkg/speech/data/FeatureTransforms.h"
#include "flashlight/lib/audio/feature/FeatureParams.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"

namespace fl {
namespace pkg {
namespace speech {

std::vector<std::string> wrd2Target(
    const std::string& word,
    const lib::text::LexiconMap& lexicon,
    const lib::text::Dictionary& dict,
    const std::string& wordSeparator = "",
    float targetSamplePct = 0,
    bool fallback2LtrWordSepLeft = false,
    bool fallback2LtrWordSepRight = false,
    bool skipUnk = false);

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const lib::text::LexiconMap& lexicon,
    const lib::text::Dictionary& dict,
    const std::string& wordSeparator = "",
    float targetSamplePct = 0,
    bool fallback2LtrWordSepLeft = false,
    bool fallback2LtrWordSepRight = false,
    bool skipUnk = false);

std::pair<int, FeatureType> getFeatureType(
    const std::string& featuresType,
    int channels,
    const fl::lib::audio::FeatureParams& featParams);

} // namespace speech
} // namespace pkg
} // namespace fl
