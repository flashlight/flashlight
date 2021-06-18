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

#include "flashlight/fl/distributed/DistributedUtils.h"
#include "flashlight/fl/contrib/modules/Transformer.h"

namespace fl {
namespace app {
namespace asr {

struct TS2SState {
  fl::Variable alpha;
  std::vector<fl::Variable> hidden;
  fl::Variable summary;
  int step;

  TS2SState() : step(0) {}
};

typedef std::shared_ptr<TS2SState> TS2SStatePtr;

class TransformerCriterion : public SequenceCriterion {
 public:
  TransformerCriterion(
      int nClass,
      int hiddenDim,
      int eos,
      int pad,
      int maxDecoderOutputLen,
      int nLayer,
      std::shared_ptr<AttentionBase> attention,
      std::shared_ptr<WindowBase> window,
      bool trainWithWindow,
      double labelSmooth,
      double pctTeacherForcing,
      double pDropout,
      double pLayerDrop);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  af::array viterbiPath(
      const af::array& input,
      const af::array& inputSizes = af::array()) override;

  std::pair<af::array, fl::Variable> viterbiPathBase(
      const af::array& input,
      const af::array& inputSizes,
      bool saveAttn);

  std::pair<fl::Variable, fl::Variable> vectorizedDecoder(
      const fl::Variable& input,
      const fl::Variable& target,
      const af::array& inputSizes,
      const af::array& targetSizes);

  std::pair<fl::Variable, TS2SState> decodeStep(
      const fl::Variable& xEncoded,
      const fl::Variable& y,
      const TS2SState& inState,
      const af::array& inputSizes) const;

  std::pair<std::vector<std::vector<float>>, std::vector<TS2SStatePtr>>
  decodeBatchStep(
      const fl::Variable& xEncoded,
      std::vector<fl::Variable>& ys,
      const std::vector<TS2SState*>& inStates,
      const int attentionThreshold,
      const float smoothingTemperature) const;

  void clearWindow() {
    trainWithWindow_ = false;
    window_ = nullptr;
  }

  std::string prettyString() const override;

  std::shared_ptr<fl::Embedding> embedding() const {
    return std::static_pointer_cast<fl::Embedding>(module(0));
  }

  std::shared_ptr<fl::Transformer> layer(int i) const {
    return std::static_pointer_cast<fl::Transformer>(module(i + 1));
  }

  std::shared_ptr<fl::Linear> linearOut() const {
    return std::static_pointer_cast<fl::Linear>(module(nLayer_ + 1));
  }

  std::shared_ptr<AttentionBase> attention() const {
    return std::static_pointer_cast<AttentionBase>(module(nLayer_ + 2));
  }

  fl::Variable startEmbedding() const {
    return params_.back();
  }

 private:
  int nClass_;
  int eos_;
  int pad_;
  int maxDecoderOutputLen_;
  int nLayer_;
  std::shared_ptr<WindowBase> window_;
  bool trainWithWindow_;
  double labelSmooth_;
  double pctTeacherForcing_;

  FL_SAVE_LOAD_WITH_BASE(
      SequenceCriterion,
      nClass_,
      eos_,
      maxDecoderOutputLen_,
      nLayer_,
      window_,
      trainWithWindow_,
      labelSmooth_,
      pctTeacherForcing_,
      fl::versioned(pad_, 1))

  TransformerCriterion() = default;
};

struct TS2SDecoderBuffer {
  fl::Variable input;
  TS2SState dummyState;
  std::vector<fl::Variable> ys;
  std::vector<TS2SState*> prevStates;
  int attentionThreshold;
  double smoothingTemperature;

  TS2SDecoderBuffer(int beamSize, int attnThre, float smootTemp)
      : attentionThreshold(attnThre), smoothingTemperature(smootTemp) {
    ys.reserve(beamSize);
    prevStates.reserve(beamSize);
  }
};

AMUpdateFunc buildSeq2SeqTransformerAmUpdateFunction(
    std::shared_ptr<SequenceCriterion>& criterion,
    int beamSize,
    float attThr,
    float smoothingTemp);
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::TransformerCriterion)
CEREAL_CLASS_VERSION(fl::app::asr::TransformerCriterion, 1)
