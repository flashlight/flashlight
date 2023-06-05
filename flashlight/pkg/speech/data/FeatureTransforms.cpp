/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/data/FeatureTransforms.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "flashlight/pkg/speech/audio/feature/Mfcc.h"
#include "flashlight/pkg/speech/audio/feature/Mfsc.h"
#include "flashlight/pkg/speech/audio/feature/PowerSpectrum.h"
#include "flashlight/lib/text/String.h"
#include "flashlight/pkg/speech/data/Utils.h"

using namespace fl::lib;
using namespace fl::lib::audio;
using fl::lib::text::Dictionary;
using fl::lib::text::LexiconMap;
using fl::lib::text::packReplabels;

namespace {

size_t getSfxSeed() {
  // A naive seed based on thread ID
  return std::hash<std::thread::id>()(std::this_thread::get_id());
}

class StartSfxCounter {
 public:
  explicit StartSfxCounter(int n) : iters_(n) {}
  bool decrementAndCheck() {
    std::lock_guard<std::mutex> lock(mutex_);
    iters_ = iters_ > 0 ? iters_ - 1 : iters_;
    return iters_ <= 0;
  }

 private:
  int iters_;
  std::mutex mutex_;
};

} // namespace

namespace fl {
namespace pkg {
namespace speech {

fl::Dataset::DataTransformFunction inputFeatures(
    const FeatureParams& params,
    const FeatureType& featureType,
    const std::pair<int, int>& localNormCtx,
    const std::vector<sfx::SoundEffectConfig>& sfxConf /* = {} */,
    const int sfxStartUpdate /* = 0 */) {
  auto sfxCounter = std::make_shared<StartSfxCounter>(sfxStartUpdate);

  std::shared_ptr<PowerSpectrum> spectralFeature;
  int featSz = 1;

  if (featureType == FeatureType::POW_SPECTRUM) {
    spectralFeature = std::make_shared<PowerSpectrum>(params);
    featSz = params.powSpecFeatSz();
  } else if (featureType == FeatureType::MFSC) {
    spectralFeature = std::make_shared<Mfsc>(params);
    featSz = params.mfscFeatSz();
  } else if (featureType == FeatureType::MFCC) {
    spectralFeature = std::make_shared<Mfcc>(params);
    featSz = params.mfccFeatSz();
  }

  return [featSz, spectralFeature, localNormCtx, sfxConf, sfxCounter](
             void* data, Shape dims, fl::dtype type) {
    if (type != fl::dtype::f32) {
      throw std::invalid_argument("Invalid input type");
    }
    if (dims.ndim() != 2) {
      throw std::invalid_argument(
          "'inputFeatures': Invalid input dims . Expected 2d array - Channels x T");
    }
    auto channels = dims[0];
    std::vector<float> input(dims.elements());
    std::copy_n(static_cast<const float*>(data), input.size(), input.data());
    if (channels > 1) {
      input = transpose2d(input, dims[1], channels);
    }
    if (!sfxConf.empty() && sfxCounter->decrementAndCheck()) {
      if (channels > 1) {
        throw std::invalid_argument(
            "'inputFeatures': Invalid input dims. sound effect supports a single channel audio");
      }
      thread_local auto seed = getSfxSeed();
      thread_local std::shared_ptr<sfx::SoundEffect> sfx =
          sfx::createSoundEffect(sfxConf, seed);
      sfx->apply(input);
    }

    std::vector<float> output;
    if (spectralFeature) {
      output = spectralFeature->batchApply(input, channels);
    } else {
      // use raw audio
      output = input; // T X CHANNELS (Col Major)
    }

    auto T = output.size() / (featSz * channels);
    // Before: FEAT X FRAMES X CHANNELS  (Col Major)
    output = transpose2d(output, T, featSz, channels);
    // After: FRAMES X FEAT X CHANNELS  (Col Major)
    if (localNormCtx.first > 0 || localNormCtx.second > 0) {
      output =
          localNormalize(output, localNormCtx.first, localNormCtx.second, T);
    } else {
      output = normalize(output);
    }
    return Tensor::fromBuffer(
        {static_cast<long long>(T), featSz, channels},
        output.data(),
        MemoryLocation::Host);
  };
}

// target
fl::Dataset::DataTransformFunction targetFeatures(
    const Dictionary& tokenDict,
    const LexiconMap& lexicon,
    const TargetGenerationConfig& config) {
  return [tokenDict, lexicon, config](
             void* data, Shape dims, fl::dtype /* unused */) {
    std::string transcript(
        static_cast<char*>(data), static_cast<char*>(data) + dims.elements());
    auto words = splitOnWhitespace(transcript, true);
    auto target = wrd2Target(
        words,
        lexicon,
        tokenDict,
        config.wordSeparator_,
        config.targetSamplePct_,
        config.fallbackToLetterWordSepLeft_,
        config.fallbackToLetterWordSepRight_,
        config.skipUnk_);
    auto tgtVec = tokenDict.mapEntriesToIndices(target);
    if (!config.surround_.empty()) {
      // add surround token at the beginning and end of target
      // only if begin/end tokens are not surround
      auto idx = tokenDict.getIndex(config.surround_);
      if (tgtVec.empty() || tgtVec.back() != idx) {
        tgtVec.emplace_back(idx);
      }
      if (tgtVec.size() > 1 && tgtVec.front() != idx) {
        tgtVec.emplace_back(idx);
        std::rotate(tgtVec.begin(), tgtVec.end() - 1, tgtVec.end());
      }
    }
    if (config.replabel_ > 0) {
      tgtVec = packReplabels(tgtVec, tokenDict, config.replabel_);
    }
    if (config.criterion_ == kAsgCriterion) {
      dedup(tgtVec);
    }
    if (config.eosToken_) {
      tgtVec.emplace_back(tokenDict.getIndex(kEosToken));
    }
    if (tgtVec.empty()) {
      // support empty target
      return Tensor(fl::dtype::s32);
    }
    return Tensor::fromVector(tgtVec);
  };
}

fl::Dataset::DataTransformFunction wordFeatures(const Dictionary& wrdDict) {
  return [wrdDict](void* data, Shape dims, fl::dtype /* unused */) {
    std::string transcript(
        static_cast<char*>(data), static_cast<char*>(data) + dims.elements());
    auto words = splitOnWhitespace(transcript, true);
    auto wrdVec = wrdDict.mapEntriesToIndices(words);
    if (wrdVec.empty()) {
      // support empty target
      return Tensor(fl::dtype::s32);
    }
    return Tensor::fromVector(wrdVec);
  };
}
} // namespace speech
} // namespace pkg
} // namespace fl
