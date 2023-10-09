/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/Seq2SeqCriterion.h"

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

namespace fl::pkg::speech {

namespace detail {
Seq2SeqState concatState(std::vector<Seq2SeqState>& stateVec) {
  if (stateVec.empty()) {
    throw std::runtime_error("Empty stateVec");
  }

  int nAttnRound = stateVec[0].hidden.size();
  Seq2SeqState newState(nAttnRound);
  newState.step = stateVec[0].step;
  newState.peakAttnPos = stateVec[0].peakAttnPos;
  newState.isValid = stateVec[0].isValid;

  std::vector<Variable> alphaVec;
  std::vector<std::vector<Variable>> hiddenVec(nAttnRound);
  std::vector<Variable> summaryVec;
  for (auto& state : stateVec) {
    if (state.step != newState.step) {
      throw std::runtime_error("step unmatched");
    } else if (state.isValid != newState.isValid) {
      throw std::runtime_error("isValid unmatched");
    }
    alphaVec.push_back(state.alpha);
    for (int i = 0; i < nAttnRound; i++) {
      hiddenVec[i].push_back(state.hidden[i]);
    }
    summaryVec.push_back(state.summary);
  }

  newState.alpha = concatenate(alphaVec, 2);
  for (int i = 0; i < nAttnRound; i++) {
    newState.hidden[i] = concatenate(hiddenVec[i], 1);
  }
  newState.summary = concatenate(summaryVec, 2);
  return newState;
}

Seq2SeqState selectState(Seq2SeqState& state, int batchIdx) {
  int nAttnRound = state.hidden.size();
  Seq2SeqState newState(nAttnRound);
  newState.step = state.step;
  newState.peakAttnPos = state.peakAttnPos;
  newState.isValid = state.isValid;
  newState.alpha =
      state.alpha(fl::span, fl::span, fl::range(batchIdx, batchIdx + 1));
  newState.summary =
      state.summary(fl::span, fl::span, fl::range(batchIdx, batchIdx + 1));
  for (int i = 0; i < nAttnRound; i++) {
    newState.hidden[i] =
        state.hidden[i](fl::span, fl::range(batchIdx, batchIdx + 1));
  }
  return newState;
}
} // namespace detail

Seq2SeqCriterion::Seq2SeqCriterion(
    int nClass,
    int hiddenDim,
    int eos,
    int pad,
    int maxDecoderOutputLen,
    const std::vector<std::shared_ptr<AttentionBase>>& attentions,
    std::shared_ptr<WindowBase> window /* = nullptr*/,
    bool trainWithWindow /* false */,
    int pctTeacherForcing /* = 100 */,
    double labelSmooth /* = 0.0 */,
    bool inputFeeding /* = false */,
    std::string samplingStrategy, /* = fl::pkg::speech::kRandSampling */
    double gumbelTemperature /* = 1.0 */,
    int nRnnLayer /* = 1 */,
    int nAttnRound /* = 1 */,
    float dropOut /* = 0.0 */)
    : eos_(eos),
      pad_(pad),
      maxDecoderOutputLen_(maxDecoderOutputLen),
      window_(window),
      trainWithWindow_(trainWithWindow),
      pctTeacherForcing_(pctTeacherForcing),
      labelSmooth_(labelSmooth),
      inputFeeding_(inputFeeding),
      nClass_(nClass),
      samplingStrategy_(samplingStrategy),
      gumbelTemperature_(gumbelTemperature),
      nAttnRound_(nAttnRound) {
  // 1. Embedding
  add(std::make_shared<Embedding>(hiddenDim, nClass_));

  // 2. RNN
  for (int i = 0; i < nAttnRound_; i++) {
    add(std::make_shared<RNN>(
        hiddenDim, hiddenDim, nRnnLayer, RnnMode::GRU, false, dropOut));
  }

  // 3. Linear
  add(std::make_shared<Linear>(hiddenDim, nClass_));
  // FIXME: Having a linear layer in between RNN and attention is only for
  // backward compatibility.

  // 4. Attention
  for (int i = 0; i < nAttnRound_; i++) {
    add(attentions[i]);
  }

  // 5. Initial hidden state
  params_.push_back(fl::uniform(Shape{hiddenDim}, -1e-1, 1e-1));
  setUseSequentialDecoder();
}

std::unique_ptr<Module> Seq2SeqCriterion::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'Seq2SeqCriterion'");
}

/**
 * Symbols describing tensor shapes used in this file:
 * - B: batch size
 * - C: number of class/tokens
 * - H: hidden dimension
 * - U: target length
 * - T: length of the time frames in the encoded X
 */

std::vector<Variable> Seq2SeqCriterion::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() < 2 || (inputs.size() > 4)) {
    throw std::invalid_argument(
        "Invalid inputs size; Seq2Seq criterion takes input, target, inputSizes [optional]");
  }
  const auto& input = inputs[0];
  const auto& target = inputs[1];
  const auto& inputSizes =
      inputs.size() == 2 ? Tensor() : inputs[2].tensor(); // 1 x B
  const auto& targetSizes =
      inputs.size() == 3 ? Tensor() : inputs[3].tensor(); // 1 x B

  Variable out, alpha;
  if (useSequentialDecoder_) {
    std::tie(out, alpha) = decoder(input, target, inputSizes, targetSizes);
  } else {
    std::tie(out, alpha) =
        vectorizedDecoder(input, target, inputSizes, targetSizes);
  }

  out = logSoftmax(out, 0); // C x U x B

  auto losses = moddims(
      sum(categoricalCrossEntropy(out, target, ReduceMode::NONE, pad_), {0}),
      {-1});
  if (train_ && labelSmooth_ > 0) {
    size_t nClass = out.dim(0);
    auto targetTiled = fl::tile(
        fl::reshape(target.tensor(), {1, target.dim(0), target.dim(1)}),
        {static_cast<long long>(nClass)});
    out = applySeq2SeqMask(out, targetTiled, pad_);
    auto smoothLoss = moddims(sum(out, {0, 1}), {-1});
    losses = (1 - labelSmooth_) * losses - (labelSmooth_ / nClass) * smoothLoss;
  }

  return {losses, out};
}

std::pair<Variable, Variable> Seq2SeqCriterion::vectorizedDecoder(
    const Variable& input,
    const Variable& target,
    const Tensor& inputSizes,
    const Tensor& targetSizes) {
  if (target.ndim() != 2) {
    throw std::invalid_argument(
        "Seq2SeqCriterion::vectorizedDecoder: "
        "target expects to be shape {U, B}");
  }
  int U = target.dim(0);
  int B = target.dim(1);
  int T = input.dim(1);

  auto hy = tile(startEmbedding(), {1, 1, B}); // H x 1 x B

  if (U > 1) {
    // Slice off eos
    auto y = target(fl::range(0, U - 1), fl::span);
    if (train_) {
      if (samplingStrategy_ == fl::pkg::speech::kModelSampling) {
        throw std::logic_error(
            "vectorizedDecoder does not support model sampling");
      } else if (samplingStrategy_ == fl::pkg::speech::kRandSampling) {
        auto mask = Variable(
            (fl::rand(y.shape()) * 100 <= pctTeacherForcing_).astype(y.type()),
            false);
        auto samples = Variable(
            (fl::rand(y.shape()) * (nClass_ - 1)).astype(y.type()), false);

        y = mask * y + (1 - mask) * samples;
      }
    }

    auto yEmbed = embedding()->forward(y);
    hy = concatenate({hy, yEmbed}, 1); // H x U x B
  }

  Variable alpha, summaries;
  for (int i = 0; i < nAttnRound_; i++) {
    hy = fl::transpose(hy, {0, 2, 1}); // H x U x B -> H x B x U
    hy = decodeRNN(i)->forward(hy);
    hy = fl::transpose(hy, {0, 2, 1}); // H x B x U ->  H x U x B

    Variable windowWeight;
    if (window_ && (!train_ || trainWithWindow_)) {
      windowWeight =
          window_->computeVectorizedWindow(U, T, B, inputSizes, targetSizes);
    }

    std::tie(alpha, summaries) = attention(i)->forward(
        hy,
        input,
        Variable(), // vectorizedDecoder does not support prev_attn input
        windowWeight,
        fl::noGrad(inputSizes));
    hy = hy + summaries;
  }

  auto out = linearOut()->forward(hy); // C x U x B
  return std::make_pair(out, alpha);
}

std::pair<Variable, Variable> Seq2SeqCriterion::decoder(
    const Variable& input,
    const Variable& target,
    const Tensor& inputSizes,
    const Tensor& targetSizes) {
  int U = target.dim(0);

  std::vector<Variable> outvec;
  std::vector<Variable> alphaVec;
  Seq2SeqState state(nAttnRound_);
  Variable y;
  for (int u = 0; u < U; u++) {
    Variable ox;
    std::tie(ox, state) =
        decodeStep(input, y, state, inputSizes, targetSizes, U);

    if (!train_) {
      y = target(fl::range(u, u + 1), fl::span);
    } else if (samplingStrategy_ == fl::pkg::speech::kGumbelSampling) {
      double eps = 1e-7;
      auto gb = -log(-log((1 - 2 * eps) * fl::rand(ox.shape()) + eps));
      ox = logSoftmax((ox + Variable(gb, false)) / gumbelTemperature_, 0);
      y = Variable(exp(ox).tensor(), false);
    } else if (fl::all(fl::rand({1}) * 100 <= fl::full({1}, pctTeacherForcing_))
                   .asScalar<bool>()) {
      y = target(fl::range(u, u + 1), fl::span);
    } else if (samplingStrategy_ == fl::pkg::speech::kModelSampling) {
      Tensor maxIdx, maxValues;
      fl::max(maxValues, maxIdx, ox.tensor(), 0);
      y = Variable(maxIdx, false);
    } else if (samplingStrategy_ == fl::pkg::speech::kRandSampling) {
      y = Variable(
          (fl::rand({1, target.dim(1)}) * (nClass_ - 1)).astype(fl::dtype::s32),
          false);
    } else {
      throw std::invalid_argument("Invalid sampling strategy");
    }

    outvec.push_back(ox);
    alphaVec.push_back(state.alpha);
  }

  auto out = concatenate(outvec, 1); // C x U x B
  auto alpha = concatenate(alphaVec, 0); // U x T x B

  return std::make_pair(out, alpha);
}

Tensor Seq2SeqCriterion::viterbiPath(
    const Tensor& input,
    const Tensor& inputSizes /* = Tensor() */) {
  return viterbiPathBase(input, inputSizes, false).first;
}

std::pair<Tensor, Variable> Seq2SeqCriterion::viterbiPathBase(
    const Tensor& input,
    const Tensor& inputSizes,
    bool saveAttn) {
  // NB: xEncoded has to be with batchsize 1
  bool wasTrain = train_;
  eval();
  std::vector<int> maxPath;
  std::vector<Variable> alphaVec;
  Variable alpha;
  Seq2SeqState state(nAttnRound_);
  Variable y, ox;
  Tensor maxIdx, maxValues;
  int pred;
  for (int u = 0; u < maxDecoderOutputLen_; u++) {
    std::tie(ox, state) = decodeStep(
        Variable(input, false), y, state, inputSizes, Tensor(), input.dim(1));
    fl::max(maxValues, maxIdx, ox.tensor(), 0);
    pred = maxIdx.asScalar<int>();
    if (saveAttn) {
      alphaVec.push_back(state.alpha);
    }

    if (pred == eos_) {
      break;
    }
    y = constant(pred, {1}, fl::dtype::s32, false);
    maxPath.push_back(pred);
  }
  if (saveAttn) {
    alpha = concatenate(alphaVec, 0);
  }

  if (wasTrain) {
    train();
  }
  Tensor vPath = maxPath.empty() ? Tensor() : Tensor::fromVector(maxPath);
  return std::make_pair(vPath, alpha);
}

std::vector<int> Seq2SeqCriterion::beamPath(
    const Tensor& input,
    const Tensor& inputSizes,
    int beamSize /* = 10 */) {
  std::vector<Seq2SeqCriterion::CandidateHypo> beam;
  beam.emplace_back();
  auto beamPaths =
      beamSearch(input, inputSizes, beam, beamSize, maxDecoderOutputLen_);
  return beamPaths[0].path;
}

// beam are candidates that need to be extended
std::vector<Seq2SeqCriterion::CandidateHypo> Seq2SeqCriterion::beamSearch(
    const Tensor& input, // H x T x 1
    const Tensor& inputSizes, // 1 x B
    std::vector<Seq2SeqCriterion::CandidateHypo> beam,
    int beamSize = 10,
    int maxLen = 200) {
  bool wasTrain = train_;
  eval();

  std::vector<Seq2SeqCriterion::CandidateHypo> complete;
  std::vector<Seq2SeqCriterion::CandidateHypo> newBeam;
  auto cmpfn = [](Seq2SeqCriterion::CandidateHypo& lhs,
                  Seq2SeqCriterion::CandidateHypo& rhs) {
    return lhs.score > rhs.score;
  };

  for (int l = 0; l < maxLen; l++) {
    newBeam.resize(0);

    std::vector<Variable> prevYVec;
    std::vector<Seq2SeqState> prevStateVec;
    std::vector<float> prevScoreVec;
    for (auto& hypo : beam) {
      Variable y;
      if (!hypo.path.empty()) {
        y = constant(hypo.path.back(), {1}, fl::dtype::s32, false);
      }
      prevYVec.push_back(y);
      prevStateVec.push_back(hypo.state);
      prevScoreVec.push_back(hypo.score);
    }
    auto prevY = concatenate(prevYVec, 1); // 1 x B
    auto prevState = detail::concatState(prevStateVec);
    int B = prevY.ndim() < 2 ? 1 : prevY.dim(1);

    Variable ox;
    Seq2SeqState state;
    // do proper cast of input size to batch size
    // because we have beam now for the input
    auto tiledInputSizes = fl::tile(inputSizes, {1, B});
    std::tie(ox, state) = decodeStep(
        Variable(input, false),
        prevY,
        prevState,
        tiledInputSizes,
        Tensor(),
        input.dim(1));
    ox = logSoftmax(ox, 0); // C x 1 x B
    ox = fl::reorder(ox, {0, 2, 1});

    auto scoreArr = Tensor::fromBuffer(
        {1, static_cast<long long>(beam.size()), 1},
        prevScoreVec.data(),
        MemoryLocation::Host);
    scoreArr = fl::tile(scoreArr, {ox.dim(0)});

    scoreArr = scoreArr + ox.tensor(); // C x B
    scoreArr = scoreArr.flatten(); // column-first
    auto scoreVec = scoreArr.toHostVector<float>();

    std::vector<size_t> indices(scoreVec.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(),
        indices.begin() +
            std::min(2 * beamSize, static_cast<int>(scoreVec.size())),
        indices.end(),
        [&scoreVec](size_t i1, size_t i2) {
          return scoreVec[i1] > scoreVec[i2];
        });

    int nClass = ox.dim(0);
    for (int j = 0; j < indices.size(); j++) {
      int hypIdx = indices[j] / nClass;
      int clsIdx = indices[j] % nClass;
      std::vector<int> path_(beam[hypIdx].path);
      path_.push_back(clsIdx);
      if (j < beamSize && clsIdx == eos_) {
        path_.pop_back();
        complete.emplace_back(
            scoreVec[indices[j]], path_, detail::selectState(state, hypIdx));
      } else if (clsIdx != eos_) {
        newBeam.emplace_back(
            scoreVec[indices[j]], path_, detail::selectState(state, hypIdx));
      }
      if (newBeam.size() >= beamSize) {
        break;
      }
    }
    beam.resize(newBeam.size());
    beam = std::move(newBeam);

    if (complete.size() >= beamSize) {
      std::partial_sort(
          complete.begin(), complete.begin() + beamSize, complete.end(), cmpfn);
      complete.resize(beamSize);

      // if lowest score in complete is better than best future hypo
      // then its not possible for any future hypothesis to replace existing
      // hypothesises in complete.
      if (complete.back().score > beam[0].score) {
        break;
      }
    }
  }

  if (wasTrain) {
    train();
  }

  return complete.empty() ? beam : complete;
}

std::pair<Variable, Seq2SeqState> Seq2SeqCriterion::decodeStep(
    const Variable& xEncoded,
    const Variable& y,
    const Seq2SeqState& inState,
    const Tensor& inputSizes,
    const Tensor& targetSizes,
    const int maxDecoderSteps) const {
  if (xEncoded.ndim() != 3) {
    throw std::invalid_argument(
        "Seq2SeqCriterion::decodeStep: "
        "expected xEncoded to have at least three dimensions");
  }

  Variable hy;
  if (y.isEmpty()) {
    hy = tile(startEmbedding(), {1, 1, static_cast<int>(xEncoded.dim(2))});
  } else if (train_ && samplingStrategy_ == fl::pkg::speech::kGumbelSampling) {
    hy = linear(y, embedding()->param(0));
  } else {
    hy = embedding()->forward(y);
  }

  if (inputFeeding_ && !y.isEmpty()) {
    hy = hy + moddims(inState.summary, hy.shape());
  }
  hy = moddims(hy, {hy.dim(0), -1}); // H x B

  Seq2SeqState outState(nAttnRound_);
  outState.step = inState.step + 1;

  Variable summaries;
  for (int i = 0; i < nAttnRound_; i++) {
    hy = moddims(hy, {hy.dim(0), -1}); // H x 1 x B -> H x B
    std::tie(hy, outState.hidden[i]) =
        decodeRNN(i)->forward(hy, inState.hidden[i]);
    hy = moddims(hy, {hy.dim(0), 1, hy.dim(1)}); // H x B -> H x 1 x B

    Variable windowWeight;
    // because of the beam search batchsize can be
    // different for xEncoded and y (xEncoded batch = 1 and y batch = beam
    // size)
    int batchsize =
        y.isEmpty() ? xEncoded.dim(2) : (y.ndim() < 2 ? 1 : y.dim(1));
    if (window_ && (!train_ || trainWithWindow_)) {
      // TODO fix for softpretrain where target size is used
      // for now force to xEncoded.dim(1)
      windowWeight = window_->computeWindow(
          inState.alpha,
          inState.step,
          maxDecoderSteps,
          xEncoded.dim(1),
          batchsize,
          inputSizes,
          targetSizes);
    }
    std::tie(outState.alpha, summaries) = attention(i)->forward(
        hy, xEncoded, inState.alpha, windowWeight, fl::noGrad(inputSizes));
    hy = hy + summaries;
  }
  outState.summary = summaries;

  auto out = linearOut()->forward(hy); // C x 1 x B
  return std::make_pair(out, outState);
}

std::pair<std::vector<std::vector<float>>, std::vector<Seq2SeqStatePtr>>
Seq2SeqCriterion::decodeBatchStep(
    const fl::Variable& xEncoded,
    std::vector<fl::Variable>& ys,
    const std::vector<Seq2SeqState*>& inStates,
    const int attentionThreshold,
    const float smoothingTemperature) const {
  // NB: xEncoded has to be with batchsize 1
  int batchSize = ys.size();
  std::vector<Variable> statesVector(batchSize);

  // Batch Ys
  for (int i = 0; i < batchSize; i++) {
    if (ys[i].isEmpty()) {
      ys[i] = startEmbedding();
    } else {
      ys[i] = embedding()->forward(ys[i]);
      if (inputFeeding_) {
        ys[i] = ys[i] + moddims(inStates[i]->summary, ys[i].shape());
      }
    }
    ys[i] = moddims(ys[i], {ys[i].dim(0), -1});
  }
  Variable yBatched = concatenate(ys, 1); // H x B

  std::vector<Seq2SeqStatePtr> outstates(batchSize);
  for (int i = 0; i < batchSize; i++) {
    outstates[i] = std::make_shared<Seq2SeqState>(nAttnRound_);
    outstates[i]->step = inStates[i]->step + 1;
  }
  Variable outStateBatched;

  for (int n = 0; n < nAttnRound_; n++) {
    /* (1) RNN forward */
    if (inStates[0]->hidden[n].isEmpty()) {
      std::tie(yBatched, outStateBatched) =
          decodeRNN(n)->forward(yBatched, Variable());
    } else {
      for (int i = 0; i < batchSize; i++) {
        statesVector[i] = inStates[i]->hidden[n];
      }
      Variable inStateHiddenBatched =
          concatenate(statesVector, 1).asContiguous();
      std::tie(yBatched, outStateBatched) =
          decodeRNN(n)->forward(yBatched, inStateHiddenBatched);
    }

    for (int i = 0; i < batchSize; i++) {
      outstates[i]->hidden[n] = outStateBatched(fl::span, fl::range(i, i + 1));
    }

    /* (2) Attention forward */
    if (window_ && (!train_ || trainWithWindow_)) {
      throw std::runtime_error(
          "Batched decoding does not support models with window");
    }

    Variable summaries, alphaBatched;
    // NB:
    // - Third Variable is set to empty since no attention use it.
    // - Only ContentAttention is supported
    std::tie(alphaBatched, summaries) =
        attention(n)->forward(yBatched, xEncoded, Variable(), Variable());
    alphaBatched = fl::transpose(alphaBatched, {1, 0, 2}); // B x T -> T x B
    yBatched = yBatched + summaries; // H x B

    Tensor bestpath, maxvalues;
    fl::max(maxvalues, bestpath, alphaBatched.tensor(), 0);
    std::vector<int> maxIdx = bestpath.toHostVector<int>();
    for (int i = 0; i < batchSize; i++) {
      outstates[i]->peakAttnPos = maxIdx[i];
      // TODO: std::abs maybe unnecessary
      outstates[i]->isValid =
          std::abs(outstates[i]->peakAttnPos - inStates[i]->peakAttnPos) <=
          attentionThreshold;
      outstates[i]->alpha = alphaBatched(fl::span, fl::range(i, i + 1));
      outstates[i]->summary = yBatched(fl::span, fl::range(i, i + 1));
    }
  }

  /* (3) Linear forward */
  auto outBatched = linearOut()->forward(yBatched);
  outBatched = logSoftmax(outBatched / smoothingTemperature, 0);
  std::vector<std::vector<float>> out(batchSize);
  for (int i = 0; i < batchSize; i++) {
    out[i] = outBatched(fl::span, fl::range(i, i + 1))
                 .tensor()
                 .toHostVector<float>();
  }

  return std::make_pair(out, outstates);
}

void Seq2SeqCriterion::setUseSequentialDecoder() {
  useSequentialDecoder_ = false;
  if ((pctTeacherForcing_ < 100 &&
       samplingStrategy_ == fl::pkg::speech::kModelSampling) ||
      samplingStrategy_ == fl::pkg::speech::kGumbelSampling || inputFeeding_) {
    useSequentialDecoder_ = true;
  } else if (
      std::dynamic_pointer_cast<SimpleLocationAttention>(attention(0)) ||
      std::dynamic_pointer_cast<LocationAttention>(attention(0)) ||
      std::dynamic_pointer_cast<NeuralLocationAttention>(attention(0))) {
    useSequentialDecoder_ = true;
  } else if (
      window_ && trainWithWindow_ &&
      std::dynamic_pointer_cast<MedianWindow>(window_)) {
    useSequentialDecoder_ = true;
  }
}

std::string Seq2SeqCriterion::prettyString() const {
  return "Seq2SeqCriterion";
}

EmittingModelUpdateFunc buildSeq2SeqRnnUpdateFunction(
    std::shared_ptr<SequenceCriterion>& criterion,
    int attRound,
    int beamSize,
    float attThr,
    float smoothingTemp) {
  auto buf = std::make_shared<Seq2SeqDecoderBuffer>(
      attRound, beamSize, attThr, smoothingTemp);

  const Seq2SeqCriterion* s2sCriterion =
      static_cast<Seq2SeqCriterion*>(criterion.get());
  auto emittingModelUpdateFunc =
      [buf, s2sCriterion](
          const float* emissions,
          const int N,
          const int T,
          const std::vector<int>& rawY,
          const std::vector<int>& /* prevHypBeamIdxs */,
          const std::vector<EmittingModelStatePtr>& rawPrevStates,
          int& t) {
        if (t == 0) {
          buf->input = fl::Variable(
              Tensor::fromBuffer({N, T}, emissions, MemoryLocation::Host),
              false);
        }
        int batchSize = rawY.size();
        buf->prevStates.resize(0);
        buf->ys.resize(0);

        // Cast to seq2seq states
        for (int i = 0; i < batchSize; i++) {
          Seq2SeqState* prevState =
              static_cast<Seq2SeqState*>(rawPrevStates[i].get());
          fl::Variable y;
          if (t > 0) {
            y = fl::constant(rawY[i], {1}, fl::dtype::s32, false);
          } else {
            prevState = &buf->dummyState;
          }
          buf->ys.push_back(y);
          buf->prevStates.push_back(prevState);
        }

        // Run forward in batch
        std::vector<std::vector<float>> amScores;
        std::vector<Seq2SeqStatePtr> outStates;

        std::tie(amScores, outStates) = s2sCriterion->decodeBatchStep(
            buf->input,
            buf->ys,
            buf->prevStates,
            buf->attentionThreshold,
            buf->smoothingTemperature);

        // Cast back to void*
        std::vector<EmittingModelStatePtr> out;
        for (auto& os : outStates) {
          if (os->isValid) {
            out.push_back(os);
          } else {
            out.push_back(nullptr);
          }
        }
        return std::make_pair(amScores, out);
      };

  return emittingModelUpdateFunc;
}

} // namespace fl
