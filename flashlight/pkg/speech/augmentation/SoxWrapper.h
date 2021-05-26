/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/augmentation/SoundEffect.h"

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"

#define FL_SOX_CHECK(expr) \
  ::fl::pkg::speech::sfx::detail::check((expr), #expr, __FILE__, __LINE__)

// Add forward declareations of sox.h related types so we can keep the include
// of sox.h in the cpp file, thus avoiding proliferation of dependency on a
// third-party library.
class sox_effect_t;
class sox_effects_chain_t;
class sox_effect_handler_t;
class sox_signalinfo_t;

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {
#ifdef FL_BUILD_APP_ASR_SFX_SOX

/**
 * Utility for creating libsox based sound effects. This is a singelton class.
 * Use example:
 * \code{.cpp}
 *  // call SoxWrapper::instance() once to initialize if sample rate!=16000.
 *  SoxWrapper::instance(sampleRate);
 *
 *  // Input signal is of type vector<float>
 *  std::vector<float> signal = loadSound<float>(filename);
 *  // User creates sox effect.
 *  sox_effect_t* e = sox_create_effect(sox_find_effect("stretch"));
 *  char* args[] = {(char*)std::to_string(factor).c_str()};
 *  FL_SOX_CHECK(sox_effect_options(e, 1, args));
 *
 *  // call applyAndFreeEffect() to stream the signal through the effect.
 *  SoxWrapper::instance()->applyAndFreeEffect(signal, e);
 * \endcode
 */
class SoxWrapper {
 public:
  static SoxWrapper* instance(size_t sampleRate = 16000);
  ~SoxWrapper();

  /**
   * Apply the given libsox effect on the signal.
   */
  void applyAndFreeEffect(std::vector<float>& signal, sox_effect_t* effect)
      const;

 private:
  explicit SoxWrapper(size_t sampleRate);

  sox_effects_chain_t* createChain() const;
  void addInput(sox_effects_chain_t* chain, std::vector<float>* signal) const;
  void addOutput(sox_effects_chain_t* chain, std::vector<float>* emptyBuf)
      const;
  void addAndFreeEffect(sox_effects_chain_t* chain, sox_effect_t* effect) const;

  // sox wants pointer to non-const sox_signalinfo_t but it does not change it.
  // mutable is so we can pass pointer to signalInfo_ from const methods.
  mutable std::unique_ptr<sox_signalinfo_t> signalInfo_;
  static std::unique_ptr<SoxWrapper> instance_;
};

#else /* ifdef FL_BUILD_APP_ASR_SFX_SOX */
// Definition with null implementation to stub out SoxWrapper
// when building sound effects without libsox.
class SoxWrapper {
 public:
  static SoxWrapper* instance(size_t sampleRate = 16000) {
    return nullptr;
  }
  void applyAndFreeEffect(std::vector<float>& signal, sox_effect_t* effect)
      const {}
};
#endif  /* FL_BUILD_APP_ASR_SFX_SOX */

namespace detail {

void check(bool success, const char* msg, const char* file, int line);
void check(int status, const char* msg, const char* file, int line);
void check(const void* ptr, const char* msg, const char* file, int line);

} // namespace detail

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
