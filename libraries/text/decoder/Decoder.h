/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/libraries/text/decoder/Utils.h"

namespace fl {
namespace lib {
namespace text {

enum class CriterionType { ASG = 0, CTC = 1, S2S = 2 };

struct DecoderOptions {
  int beamSize; // Maximum number of hypothesis we hold after each step
  int beamSizeToken; // Maximum number of tokens we consider at each step
  double beamThreshold; // Threshold to prune hypothesis
  double lmWeight; // Weight of lm
  double wordScore; // Word insertion score
  double unkScore; // Unknown word insertion score
  double silScore; // Silence insertion score
  double eosScore; // Score for inserting an EOS
  bool logAdd; // If or not use logadd when merging hypothesis
  CriterionType criterionType; // CTC or ASG

  DecoderOptions(
      const int beamSize,
      const int beamSizeToken,
      const double beamThreshold,
      const double lmWeight,
      const double wordScore,
      const double unkScore,
      const double silScore,
      const double eosScore,
      const bool logAdd,
      const CriterionType criterionType)
      : beamSize(beamSize),
        beamSizeToken(beamSizeToken),
        beamThreshold(beamThreshold),
        lmWeight(lmWeight),
        wordScore(wordScore),
        unkScore(unkScore),
        silScore(silScore),
        eosScore(eosScore),
        logAdd(logAdd),
        criterionType(criterionType) {}

  DecoderOptions() {}
};

/**
 * Decoder support two typical use cases:
 * Offline manner:
 *  decoder.decode(someData) [returns all hypothesis (transcription)]
 *
 * Online manner:
 *  decoder.decodeBegin() [called only at the beginning of the stream]
 *  while (stream)
 *    decoder.decodeStep(someData) [one or more calls]
 *    decoder.getBestHypothesis() [returns the best hypothesis (transcription)]
 *    decoder.prune() [prunes the hypothesis space]
 *  decoder.decodeEnd() [called only at the end of the stream]
 *
 * Note: function decoder.prune() deletes hypothesis up until time when called
 * to supports online decoding. It will also add a offset to the scores in beam
 * to avoid underflow/overflow.
 *
 */
class Decoder {
 public:
  explicit Decoder(const DecoderOptions& opt) : opt_(opt) {}
  virtual ~Decoder() = default;

  /* Initialize decoder before starting consume emissions */
  virtual void decodeBegin() {}

  /* Consume emissions in T x N chunks and increase the hypothesis space */
  virtual void decodeStep(const float* emissions, int T, int N) = 0;

  /* Finish up decoding after consuming all emissions */
  virtual void decodeEnd() {}

  /* Offline decode function, which consume all emissions at once */
  virtual std::vector<DecodeResult>
  decode(const float* emissions, int T, int N) {
    decodeBegin();
    decodeStep(emissions, T, N);
    decodeEnd();
    return getAllFinalHypothesis();
  }

  /* Prune the hypothesis space */
  virtual void prune(int lookBack = 0) = 0;

  /* Get the number of decoded frame in buffer */
  virtual int nDecodedFramesInBuffer() const = 0;

  /*
   * Get the best completed hypothesis which is `lookBack` frames ahead the last
   * one in buffer. For lexicon requiredd LMs, completed hypothesis means no
   * partial word appears at the end.
   */
  virtual DecodeResult getBestHypothesis(int lookBack = 0) const = 0;

  /* Get all the final hypothesis */
  virtual std::vector<DecodeResult> getAllFinalHypothesis() const = 0;

 protected:
  DecoderOptions opt_;
};
} // namespace text
} // namespace lib
} // namespace fl
