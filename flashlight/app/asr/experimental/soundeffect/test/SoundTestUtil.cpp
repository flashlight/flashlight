/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/test/SoundTestUtil.h"

#include <cmath>
#include <memory>

#include <arrayfire.h>
#include <flashlight/fl/flashlight.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "flashlight/app/asr/experimental/soundeffect/SoundEffect.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundLoader.h"
#include "flashlight/lib/common/System.h"

using ::fl::app::asr::sfx::Sound;
using ::fl::app::asr::sfx::SoundEffect;
using ::fl::app::asr::sfx::SoundLoader;
using ::fl::lib::pathsConcat;

namespace fl {
namespace app {
namespace asr {
namespace sfx {

namespace {
std::string loadPath = "/tmp";
}

Sound genSinWave(
    size_t numSamples,
    size_t freq,
    size_t sampleRate,
    float amplitude) {
  auto output = std::make_shared<std::vector<float>>(numSamples, 0);
  const float waveLenSamples =
      static_cast<float>(sampleRate) / static_cast<float>(freq);
  const float ratio = (2 * M_PI) / waveLenSamples;

  for (size_t i = 0; i < numSamples; ++i) {
    output->at(i) = amplitude * std::sin(static_cast<float>(i) * ratio);
  }
  return Sound(output);
}

std::string testFilename(const std::string& filename) {
  return pathsConcat(
      pathsConcat(
          loadPath,
          ::testing::UnitTest::GetInstance()->current_test_info()->name()),
      filename);
}

std::vector<std::pair<std::string, Sound>> writeSinWaveSoundFiles(
    const std::string& basedir,
    int nSounds,
    const size_t len,
    const float amplitude,
    size_t freqStart,
    size_t freqEnd) {
  std::vector<std::pair<std::string, Sound>> filenameAndSounds;

  for (int i = 0; i < nSounds; ++i) {
    const size_t freq = freqStart + ((freqEnd - freqStart) * i) / nSounds;
    Sound sound = genSinWave(len, freq, /*sampleRate=*/16000, amplitude);
    std::stringstream ss;
    ss << "len-" << len << "-freq-" << freq << "-amp-" << amplitude << ".flac";
    sound.listFileEntry_.audioFilePath_ = pathsConcat(basedir, ss.str());
    try {
      sound.writeToFile();
    } catch (std::exception& ex) {
      std::cerr << "writeSinWaveSoundFiles(basedir=" << basedir
                << " nSounds=" << nSounds << ") failed to save file="
                << sound.listFileEntry_.audioFilePath_
                << " with error=" << ex.what() << std::endl;
      continue;
    }
    filenameAndSounds.push_back({sound.listFileEntry_.audioFilePath_, sound});
  }

  return filenameAndSounds;
}

std::vector<sfx::ListFileEntry> writeListFile(
    const std::string filename,
    const std::vector<std::pair<std::string, Sound>>& filenamesAndSounds) {
  const int nFiles = 100;

  std::vector<sfx::ListFileEntry> listFileEntries;
  {
    sfx::ListFileWriter writer(filename);
    std::cout << "writer=" << writer.prettyString() << std::endl;
    for (int i = 0; i < filenamesAndSounds.size(); ++i) {
      sfx::ListFileEntry entry;
      entry.sampleId_ = std::to_string(i);
      entry.audioFilePath_ = filenamesAndSounds[i].first;
      entry.audioSize_ = filenamesAndSounds[i].second.size();
      entry.transcript_ = {"the good the bad and the ugly"};

      writer.write(entry);
      listFileEntries.push_back(entry);
    }
  }

  return listFileEntries;
}

void debugPrintSound(
    const std::string name,
    Sound sound,
    const Interval& interval,
    int mark1,
    int mark2) {
  const double debugWaveAmplitude = 50;
  const int debugNumFrame = 160;

  Interval intersection = intervalIntersection({interval, {0, sound.size()}});
  intersection = intervalIntersection(
      {intersection, {intersection.first, intersection.first + debugNumFrame}});

  const std::vector<float>& data = *sound.getCpuData();

  std::cout << name << ":" << std::endl;
  for (int i = intersection.first; i < intersection.second; ++i) {
    const char ch = (i == mark1) ? 'X' : ((i == mark2) ? '$' : '*');

    std::cout << std::setw(3) << i << ' ' << std::setw(6)
              << std::setprecision(4) << data[i]
              << std::setw(
                     debugWaveAmplitude * 2 + data[i] * debugWaveAmplitude)
              << ch << std::endl;
  }
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
