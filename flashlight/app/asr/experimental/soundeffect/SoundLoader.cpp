/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/SoundLoader.h"

#include <dirent.h>
#include <sys/types.h>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>

#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"
#include "flashlight/fl/common/CppBackports.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

SoundLoader::SoundLoader(ListFileReader listFileReader)
    : listFileReader_(std::move(listFileReader)) {}

size_t SoundLoader::size() {
  return listFileReader_.size();
}

Sound SoundLoader::loadIndex(int index) {
  return Sound::readFromFile(listFileReader_.read(index));
}

std::string SoundLoader::prettyString() const {
  std::stringstream ss;
  ss << "listFileReader_={" << listFileReader_.prettyString()
     << "} device=" << af::getDevice();
  return ss.str();
}

Sound SoundLoader::loadRandom() {
  return loadRandomImpl();
}

SoundLoaderRandomWithReplacement::SoundLoaderRandomWithReplacement(
    ListFileReader listFileReader,
    unsigned int randomSeed)
    : SoundLoader(std::move(listFileReader)),
      randomEngine_(randomSeed),
      randomIndex_(0, SoundLoader::size() - 1) {
  FL_LOG(fl::INFO) << "Created " << prettyString();
}

Sound SoundLoaderRandomWithReplacement::loadRandomImpl() {
  int index = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    index = randomIndex_(randomEngine_);
  }
  return loadIndex(index);
}

std::string SoundLoaderRandomWithReplacement::prettyString() const {
  std::stringstream ss;
  ss << "SoundLoaderRandomWithReplacement{SoundLoader={"
     << SoundLoader::prettyString() << "}}";
  return ss.str();
}

SoundLoaderRandomWithoutReplacement::SoundLoaderRandomWithoutReplacement(
    ListFileReader listFileReader,
    unsigned int randomSeed)
    : SoundLoader(listFileReader),
      indexShuffle_(size(), 0),
      randomReadCount_(0) {
  std::iota(indexShuffle_.begin(), indexShuffle_.end(), 0);
  std::shuffle(indexShuffle_.begin(), indexShuffle_.end(), randomEngine_);
  FL_LOG(fl::INFO) << "Created " << prettyString();
};

std::string SoundLoaderRandomWithoutReplacement::prettyString() const {
  std::stringstream ss;
  ss << "SoundLoaderRandomWithoutReplacement{randomReadCount_="
     << randomReadCount_ << " SoundLoader={" << SoundLoader::prettyString()
     << "}}";
  return ss.str();
}

Sound SoundLoaderRandomWithoutReplacement::loadRandomImpl() {
  int index = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    index = randomReadCount_++;
  }
  return loadIndex(indexShuffle_[index % indexShuffle_.size()]);
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
