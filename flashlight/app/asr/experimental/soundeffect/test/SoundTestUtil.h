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

#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

Sound genSinWave(
    size_t numSamples,
    size_t freq,
    size_t sampleRate,
    float amplitude);

std::vector<std::pair<std::string, Sound>> writeSinWaveSoundFiles(
    const std::string& basedir,
    int nSounds,
    const size_t len,
    const float amplitude,
    size_t freqStart,
    size_t freqEnd);

std::vector<ListFileEntry> writeListFile(
    const std::string filename,
    const std::vector<std::pair<std::string, Sound>>& filenamesAndSounds);

std::vector<float> vectorDiff(
    const std::vector<float>& a,
    const std::vector<float>& b);

std::string testFilename(const std::string& filename);

void debugPrintSound(
    const std::string name,
    Sound sound,
    const Interval& interval,
    int mark1 = -1,
    int mark2 = -1);

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
