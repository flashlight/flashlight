/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/pkg/speech/criterion/Defines.h"
#include "flashlight/pkg/speech/criterion/SequenceCriterion.h"
#include "flashlight/pkg/speech/criterion/attention/attention.h"
#include "flashlight/pkg/speech/criterion/attention/window.h"

#include "flashlight/ext/common/DistributedUtils.h"

namespace fl {
namespace app {
namespace asr {

struct Seq2SeqState {
  fl::Variable alpha;
  std::vector<fl::Variable> hidden;
  fl::Variable summary;
  int step;
  int peakAttnPos;
  bool isValid;

  Seq2SeqState() : hidden(1), step(0), peakAttnPos(-1), isValid(false) {}

  explicit Seq2SeqState(int nAttnRound)
      : hidden(nAttnRound), step(0), peakAttnPos(-1), isValid(false) {}
};

typedef std::shared_ptr<Seq2SeqState> Seq2SeqStatePtr;

class Seq2SeqCriterion : public SequenceCriterion {
 public:
  struct CandidateHypo {
    float score;
    std::vector<int> path;
    Seq2SeqState state;
    explicit CandidateHypo() : score(0.0) {
      path.resize(0);
    }
    CandidateHypo(float score_, std::vector<int> path_, Seq2SeqState state_)
        : score(score_), path(path_), state(state_) {}
  };

  Seq2SeqCriterion(
      int nClass,
      int hiddenDim,
      int eos,
      int pad,
      int maxDecoderOutputLen,
      const std::vector<std::shared_ptr<AttentionBase>>& attentions,
      std::shared_ptr<WindowBase> window = nullptr,
      bool trainWithWindow = false,
      int pctTeacherForcing = 100,
      double labelSmooth = 0.0,
      bool inputFeeding = false,
      std::string samplingStrategy = fl::app::asr::kRandSampling,
      double gumbelTemperature = 1.0,
      int nRnnLayer = 1,
      int nAttnRound = 1,
      float dropOut = 0.0);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  /* Next step predictions are based on the target at
   * the previous time-step so this function should only
   * be used for training purposes. */
  std::pair<fl::Variable, fl::Variable> decoder(
      const fl::Variable& input,
      const fl::Variable& target,
      const af::array& inputSizes,
      const af::array& targetSizes);

  std::pair<fl::Variable, fl::Variable> vectorizedDecoder(
      const fl::Variable& input,
      const fl::Variable& target,
      const af::array& inputSizes,
      const af::array& targetSizes);

  af::array viterbiPath(
      const af::array& input,
      const af::array& inputSizes = af::array()) override;

  std::pair<af::array, fl::Variable> viterbiPathBase(
      const af::array& input,
      const af::array& inputSizes,
      bool saveAttn);

  std::vector<CandidateHypo> beamSearch(
      const af::array& input,
      const af::array& inputSizes,
      std::vector<Seq2SeqCriterion::CandidateHypo> beam,
      int beamSize,
      int maxLen);

  std::vector<int> beamPath(
      const af::array& input,
      const af::array& inputSizes,
      int beamSize = 10);

  std::string prettyString() const override;

  std::shared_ptr<fl::Embedding> embedding() const {
    return std::static_pointer_cast<fl::Embedding>(module(0));
  }

  std::shared_ptr<fl::RNN> decodeRNN(int n) const {
    return std::static_pointer_cast<fl::RNN>(module(n + 1));
  }

  std::shared_ptr<AttentionBase> attention(int n) const {
    return std::static_pointer_cast<AttentionBase>(module(nAttnRound_ + n + 2));
  }

  std::shared_ptr<fl::Linear> linearOut() const {
    return std::static_pointer_cast<fl::Linear>(module(nAttnRound_ + 1));
  }

  fl::Variable startEmbedding() const {
    return params_.back();
  }

  std::pair<std::vector<std::vector<float>>, std::vector<Seq2SeqStatePtr>>
  decodeBatchStep(
      const fl::Variable& xEncoded,
      std::vector<fl::Variable>& ys,
      const std::vector<Seq2SeqState*>& inStates,
      const int attentionThreshold = std::numeric_limits<int>::infinity(),
      const float smoothingTemperature = 1.0) const;

  std::pair<fl::Variable, Seq2SeqState> decodeStep(
      const fl::Variable& xEncoded,
      const fl::Variable& y,
      const Seq2SeqState& instate,
      const af::array& inputSizes,
      const af::array& targetSizes,
      int targetLen) const;

  void clearWindow() {
    trainWithWindow_ = false;
    window_ = nullptr;
  }

  void setSampling(std::string newSamplingStrategy, int newPctTeacherForcing) {
    pctTeacherForcing_ = newPctTeacherForcing;
    samplingStrategy_ = newSamplingStrategy;
    setUseSequentialDecoder();
  }

  void setGumbelTemperature(double temperature) {
    gumbelTemperature_ = temperature;
  }

  void setLabelSmooth(double labelSmooth) {
    labelSmooth_ = labelSmooth;
  }

 private:
  int eos_;
  int pad_;
  int maxDecoderOutputLen_;
  std::shared_ptr<WindowBase> window_;
  bool trainWithWindow_;
  int pctTeacherForcing_;
  bool useSequentialDecoder_;
  double labelSmooth_;
  bool inputFeeding_;
  int nClass_;
  std::string samplingStrategy_;
  double gumbelTemperature_;
  int nAttnRound_{1};

  FL_SAVE_LOAD_WITH_BASE(
      SequenceCriterion,
      eos_,
      maxDecoderOutputLen_,
      window_,
      trainWithWindow_,
      pctTeacherForcing_,
      useSequentialDecoder_,
      labelSmooth_,
      inputFeeding_,
      nClass_,
      fl::versioned(samplingStrategy_, 1),
      fl::versioned(gumbelTemperature_, 2),
      fl::versioned(nAttnRound_, 3),
      fl::versioned(pad_, 4))

  Seq2SeqCriterion() = default;

  void setUseSequentialDecoder();
};

/* Decoder helpers */
struct Seq2SeqDecoderBuffer {
  fl::Variable input;
  Seq2SeqState dummyState;
  std::vector<fl::Variable> ys;
  std::vector<Seq2SeqState*> prevStates;
  int attentionThreshold;
  double smoothingTemperature;

  Seq2SeqDecoderBuffer(
      int nAttnRound,
      int beamSize,
      int attnThre,
      int smootTemp)
      : dummyState(nAttnRound),
        attentionThreshold(attnThre),
        smoothingTemperature(smootTemp) {
    ys.reserve(beamSize);
    prevStates.reserve(beamSize);
  }
};

AMUpdateFunc buildSeq2SeqRnnAmUpdateFunction(
    std::shared_ptr<SequenceCriterion>& criterion,
    int attRound,
    int beamSize,
    float attThr,
    float smoothingTemp);
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::Seq2SeqCriterion)
CEREAL_CLASS_VERSION(fl::app::asr::Seq2SeqCriterion, 3)
