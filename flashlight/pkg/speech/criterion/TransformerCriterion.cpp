/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/TransformerCriterion.h"

#include <algorithm>
#include <queue>

#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

using namespace fl;

namespace fl {
namespace pkg {
namespace speech {

TransformerCriterion::TransformerCriterion(
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
    double pLayerDrop)
    : nClass_(nClass),
      eos_(eos),
      pad_(pad),
      maxDecoderOutputLen_(maxDecoderOutputLen),
      nLayer_(nLayer),
      window_(window),
      trainWithWindow_(trainWithWindow),
      labelSmooth_(labelSmooth),
      pctTeacherForcing_(pctTeacherForcing) {
  add(std::make_shared<fl::Embedding>(hiddenDim, nClass));
  for (size_t i = 0; i < nLayer_; i++) {
    add(std::make_shared<Transformer>(
        hiddenDim,
        hiddenDim / 4,
        hiddenDim * 4,
        4,
        maxDecoderOutputLen,
        pDropout,
        pLayerDrop,
        true));
  }
  add(std::make_shared<fl::Linear>(hiddenDim, nClass));
  add(attention);
  params_.push_back(fl::uniform(af::dim4{hiddenDim}, -1e-1, 1e-1));
}

std::vector<Variable> TransformerCriterion::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() < 2 || inputs.size() > 4) {
    throw std::invalid_argument(
        "Invalid inputs size; Transformer criterion takes input,"
        " target, inputSizes [optional], targetSizes [optional]");
  }
  const auto& input = inputs[0];
  const auto& target = inputs[1];
  const auto& inputSizes =
      inputs.size() == 2 ? af::array() : inputs[2].array(); // 1 x B
  const auto& targetSizes =
      inputs.size() == 3 ? af::array() : inputs[3].array(); // 1 x B

  Variable out, alpha;
  std::tie(out, alpha) =
      vectorizedDecoder(input, target, inputSizes, targetSizes);

  out = logSoftmax(out, 0);

  auto losses = moddims(
      sum(categoricalCrossEntropy(out, target, ReduceMode::NONE, pad_), {0}),
      -1);
  if (train_ && labelSmooth_ > 0) {
    size_t nClass = out.dims(0);
    auto targetTiled = af::tile(
        af::moddims(
            target.array(), af::dim4(1, target.dims(0), target.dims(1))),
        nClass);
    out = applySeq2SeqMask(out, targetTiled, pad_);
    auto smoothLoss = moddims(sum(out, {0, 1}), -1);
    losses = (1 - labelSmooth_) * losses - (labelSmooth_ / nClass) * smoothLoss;
  }

  return {losses, out};
}

// input : D x T x B
// target: U x B
std::pair<Variable, Variable> TransformerCriterion::vectorizedDecoder(
    const Variable& input,
    const Variable& target,
    const af::array& inputSizes,
    const af::array& targetSizes) {
  int U = target.dims(0);
  int B = target.dims(1);
  int T = input.isempty() ? 0 : input.dims(1);

  auto hy = tile(startEmbedding(), {1, 1, B});

  if (U > 1) {
    auto y = target(af::seq(0, U - 2), af::span);

    if (train_) {
      // TODO: other sampling strategies
      auto mask = Variable(
          (af::randu(y.dims()) * 100 <= pctTeacherForcing_).as(y.type()),
          false);
      auto samples =
          Variable((af::randu(y.dims()) * (nClass_ - 1)).as(y.type()), false);

      y = mask * y + (1 - mask) * samples;
    }

    auto yEmbed = embedding()->forward(y);
    hy = concatenate({hy, yEmbed}, 1);
  }

  Variable alpha, summaries;
  Variable padMask; // no mask, decoder is not looking into future
  for (int i = 0; i < nLayer_; i++) {
    hy = layer(i)->forward(std::vector<Variable>({hy, padMask})).front();
  }

  if (!input.isempty()) {
    Variable windowWeight;
    if (window_ && (!train_ || trainWithWindow_)) {
      windowWeight =
          window_->computeVectorizedWindow(U, T, B, inputSizes, targetSizes);
    }

    std::tie(alpha, summaries) = attention()->forward(
        hy, input, Variable(), windowWeight, fl::noGrad(inputSizes));

    hy = hy + summaries;
  }

  auto out = linearOut()->forward(hy);

  return std::make_pair(out, alpha);
}

af::array TransformerCriterion::viterbiPath(
    const af::array& input,
    const af::array& inputSizes /* = af::array() */) {
  return viterbiPathBase(input, inputSizes, false).first;
}

std::pair<af::array, Variable> TransformerCriterion::viterbiPathBase(
    const af::array& input,
    const af::array& inputSizes,
    bool /* TODO: saveAttn */) {
  bool wasTrain = train_;
  eval();
  std::vector<int> path;
  std::vector<Variable> alphaVec;
  Variable alpha;
  TS2SState state;
  Variable y, ox;
  af::array maxIdx, maxValues;
  int pred;

  for (int u = 0; u < maxDecoderOutputLen_; u++) {
    std::tie(ox, state) =
        decodeStep(Variable(input, false), y, state, inputSizes);
    max(maxValues, maxIdx, ox.array());
    maxIdx.host(&pred);
    // TODO: saveAttn

    if (pred == eos_) {
      break;
    }
    y = constant(pred, 1, s32, false);
    path.push_back(pred);
  }
  // TODO: saveAttn

  if (wasTrain) {
    train();
  }

  auto vPath = path.empty() ? af::array() : af::array(path.size(), path.data());
  return std::make_pair(vPath, alpha);
}

std::pair<Variable, TS2SState> TransformerCriterion::decodeStep(
    const Variable& xEncoded,
    const Variable& y,
    const TS2SState& inState,
    const af::array& inputSizes) const {
  Variable hy;
  if (y.isempty()) {
    hy = tile(startEmbedding(), {1, 1, xEncoded.dims(2)});
  } else {
    hy = embedding()->forward(y);
  }

  // TODO: inputFeeding

  TS2SState outState;
  outState.step = inState.step + 1;
  af::array padMask; // no mask because we are doing step by step decoding here,
                     // no look in the future
  for (int i = 0; i < nLayer_; i++) {
    if (inState.step == 0) {
      outState.hidden.push_back(hy);
      hy = layer(i)
               ->forward(std::vector<Variable>({hy, fl::noGrad(padMask)}))
               .front();
    } else {
      outState.hidden.push_back(concatenate({inState.hidden[i], hy}, 1));
      hy = layer(i)
               ->forward({inState.hidden[i], hy, fl::noGrad(padMask)})
               .front();
    }
  }

  Variable windowWeight, alpha, summary;
  if (window_ && (!train_ || trainWithWindow_)) {
    // TODO fix for softpretrain where target size is used
    // for now force to xEncoded.dims(1)
    windowWeight = window_->computeWindow(
        Variable(),
        inState.step,
        xEncoded.dims(1),
        xEncoded.dims(1),
        xEncoded.dims(2),
        inputSizes,
        af::array());
  }

  std::tie(alpha, summary) = attention()->forward(
      hy, xEncoded, Variable(), windowWeight, fl::noGrad(inputSizes));

  hy = hy + summary;

  auto out = linearOut()->forward(hy);
  return std::make_pair(out, outState);
}

std::pair<std::vector<std::vector<float>>, std::vector<TS2SStatePtr>>
TransformerCriterion::decodeBatchStep(
    const fl::Variable& xEncoded,
    std::vector<fl::Variable>& ys,
    const std::vector<TS2SState*>& inStates,
    const int /* attentionThreshold */,
    const float smoothingTemperature) const {
  // assume xEncoded has batch 1
  int B = ys.size();

  for (int i = 0; i < B; i++) {
    if (ys[i].isempty()) {
      ys[i] = startEmbedding();
    } else {
      ys[i] = embedding()->forward(ys[i]);
    } // TODO: input feeding
    ys[i] = moddims(ys[i], {ys[i].dims(0), 1, -1});
  }
  Variable yBatched = concatenate(ys, 2); // D x 1 x B

  std::vector<TS2SStatePtr> outstates(B);
  for (int i = 0; i < B; i++) {
    outstates[i] = std::make_shared<TS2SState>();
    outstates[i]->step = inStates[i]->step + 1;
  }

  Variable outStateBatched;
  for (int i = 0; i < nLayer_; i++) {
    if (inStates[0]->step == 0) {
      for (int j = 0; j < B; j++) {
        outstates[j]->hidden.push_back(yBatched.slice(j));
      }
      yBatched = layer(i)->forward(std::vector<Variable>({yBatched})).front();
    } else {
      std::vector<Variable> statesVector(B);
      for (int j = 0; j < B; j++) {
        statesVector[j] = inStates[j]->hidden[i];
      }
      Variable inStateHiddenBatched = concatenate(statesVector, 2);
      auto tmp = std::vector<Variable>({inStateHiddenBatched, yBatched});
      auto tmp2 = concatenate(tmp, 1);
      for (int j = 0; j < B; j++) {
        outstates[j]->hidden.push_back(tmp2.slice(j));
      }
      yBatched = layer(i)->forward(tmp).front();
    }
  }

  Variable alpha, summary;
  yBatched = moddims(yBatched, {yBatched.dims(0), -1});
  std::tie(alpha, summary) =
      attention()->forward(yBatched, xEncoded, Variable(), Variable());
  alpha = reorder(alpha, 1, 0);
  yBatched = yBatched + summary;

  auto outBatched = linearOut()->forward(yBatched);
  outBatched = logSoftmax(outBatched / smoothingTemperature, 0);
  std::vector<std::vector<float>> out(B);
  for (int i = 0; i < B; i++) {
    out[i] = afToVector<float>(outBatched.col(i));
  }

  return std::make_pair(out, outstates);
}

AMUpdateFunc buildSeq2SeqTransformerAmUpdateFunction(
    std::shared_ptr<SequenceCriterion>& criterion,
    int beamSize,
    float attThr,
    float smoothingTemp) {
  auto buf =
      std::make_shared<TS2SDecoderBuffer>(beamSize, attThr, smoothingTemp);

  const TransformerCriterion* criterionCast =
      static_cast<TransformerCriterion*>(criterion.get());

  auto amUpdateFunc = [buf, criterionCast](
                          const float* emissions,
                          const int N,
                          const int T,
                          const std::vector<int>& rawY,
                          const std::vector<AMStatePtr>& rawPrevStates,
                          int& t) {
    if (t == 0) {
      buf->input = fl::Variable(af::array(N, T, emissions), false);
    }
    int B = rawY.size();
    std::vector<AMStatePtr> out;
    std::vector<std::vector<float>> amScoresAll;

    // Store the latest index of the hidden state when we can clear it
    std::map<TS2SState*, int> lastIndexOfStatePtr;
    for (int index = 0; index < rawPrevStates.size(); index++) {
      TS2SState* ptr = static_cast<TS2SState*>(rawPrevStates[index].get());
      lastIndexOfStatePtr[ptr] = index;
    }

    int start = 0, step = std::min(10, 1000 / (t + 1));
    while (start < B) {
      buf->prevStates.resize(0);
      buf->ys.resize(0);

      int end = start + step;
      if (end > B) {
        end = B;
      }
      for (int i = start; i < end; i++) {
        TS2SState* prevState = static_cast<TS2SState*>(rawPrevStates[i].get());
        fl::Variable y;
        if (t > 0) {
          y = fl::constant(rawY[i], 1, s32, false);
        } else {
          prevState = &buf->dummyState;
        }
        buf->ys.push_back(y);
        buf->prevStates.push_back(prevState);
      }
      std::vector<std::vector<float>> amScores;
      std::vector<TS2SStatePtr> outStates;
      std::tie(amScores, outStates) = criterionCast->decodeBatchStep(
          buf->input,
          buf->ys,
          buf->prevStates,
          buf->attentionThreshold,
          buf->smoothingTemperature);
      for (auto& os : outStates) {
        out.push_back(os);
      }
      for (auto& s : amScores) {
        amScoresAll.push_back(s);
      }
      // clean the previous state which is not needed anymore
      // to prevent from OOM
      for (int i = start; i < end; i++) {
        TS2SState* prevState = static_cast<TS2SState*>(rawPrevStates[i].get());
        if (prevState &&
            (lastIndexOfStatePtr.find(prevState) == lastIndexOfStatePtr.end() ||
             lastIndexOfStatePtr.find(prevState)->second == i)) {
          prevState->hidden.clear();
        }
      }
      start += step;
    }
    return std::make_pair(amScoresAll, out);
  };

  return amUpdateFunc;
}

std::string TransformerCriterion::prettyString() const {
  return "TransformerCriterion";
}
} // namespace speech
} // namespace pkg
} // namespace fl
