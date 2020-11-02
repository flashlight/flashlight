/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

/**
 * Abstract sound loader base class.
 */
class SoundLoader {
 public:
  explicit SoundLoader(ListFileReader listFileReader);
  virtual ~SoundLoader() = default;
  size_t size();
  Sound loadIndex(int index);
  Sound loadRandom();
  virtual Sound loadRandomImpl() = 0;
  virtual std::string prettyString() const;

 private:
  ListFileReader listFileReader_;
};

/**
 * Random with replacment sound loader.
 * Thread safe.
 */
class SoundLoaderRandomWithReplacement : public SoundLoader {
 public:
  explicit SoundLoaderRandomWithReplacement(
      ListFileReader listFileReader,
      unsigned int randomSeed = std::mt19937::default_seed);
  ~SoundLoaderRandomWithReplacement() override = default;
  Sound loadRandomImpl() override;
  std::string prettyString() const;

 private:
  std::mutex mutex_;
  std::mt19937 randomEngine_;
  std::uniform_int_distribution<> randomIndex_;
};

/**
 * Random without replacment sound loader.
 * Thread safe.
 */
class SoundLoaderRandomWithoutReplacement : public SoundLoader {
 public:
  explicit SoundLoaderRandomWithoutReplacement(
      ListFileReader listFileReader,
      unsigned int randomSeed = std::mt19937::default_seed);
  ~SoundLoaderRandomWithoutReplacement() override = default;
  Sound loadRandomImpl() override;
  std::string prettyString() const;

 private:
  std::mutex mutex_;
  std::mt19937 randomEngine_;
  std::vector<int> indexShuffle_;
  int randomReadCount_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
