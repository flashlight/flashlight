/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/flashlight.h"

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/criterion/attention/attention.h"
#include "flashlight/pkg/speech/criterion/criterion.h"

using namespace fl;
using namespace fl::pkg::speech;

TEST(Seq2SeqTest, Seq2Seq) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "RNN gradient computation not supported for CPU backend";
  }
  int nclass = 40;
  int hiddendim = 256;
  int batchsize = 2;
  int inputsteps = 200;
  int outputsteps = 50;
  int maxoutputlen = 100;
  // int nAttnRound = 2;
  int nAttnRound = 1;

  std::vector<std::shared_ptr<AttentionBase>> attentions(
      nAttnRound, std::make_shared<ContentAttention>());
  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 2 /* eos token index */,
      nclass - 1 /* pad token index */,
      maxoutputlen,
      attentions,
      nullptr,
      false,
      100,
      0.0,
      false,
      kRandSampling,
      1.0,
      2, // nRnnLayer
      nAttnRound,
      0.0);

  auto input = fl::randn({hiddendim, inputsteps, batchsize}, fl::dtype::f32);
  auto target =
      fl::rand({outputsteps, batchsize}, fl::dtype::f32) * 0.99 * nclass;
  target = target.astype(fl::dtype::s32);

  Variable output, attention;
  std::tie(output, attention) = seq2seq.vectorizedDecoder(
      noGrad(input), noGrad(target), Tensor(), Tensor());

  ASSERT_EQ(output.shape(), Shape({nclass, outputsteps, batchsize}));

  ASSERT_EQ(attention.shape(), Shape({outputsteps, inputsteps, batchsize}));

  auto losses =
      seq2seq({fl::noGrad(input), fl::noGrad(target), fl::noGrad(Tensor())})
          .front();
  ASSERT_EQ(losses.dim(0), batchsize);

  // Backward runs.
  losses.backward();

  // Check that vecotrized decoder and sequential decoder give the same
  // results.
  Variable outSeq, attentionSeq;
  std::tie(outSeq, attentionSeq) =
      seq2seq.decoder(noGrad(input), noGrad(target), Tensor(), Tensor());

  ASSERT_TRUE(allClose(output, outSeq, 1e-6));
  ASSERT_TRUE(allClose(attention, attentionSeq, 1e-6));

  // Check size 1 Target works
  target = target(fl::range(0, 1), fl::span);
  auto loss =
      seq2seq({noGrad(input), noGrad(target), fl::noGrad(Tensor())}).front();

  // Make sure eval mode is not storing variables.
  seq2seq.eval();
  std::tie(outSeq, attentionSeq) =
      seq2seq.decoder(noGrad(input), noGrad(target), Tensor(), Tensor());
  ASSERT_FALSE(outSeq.isCalcGrad());
  ASSERT_FALSE(attentionSeq.isCalcGrad());
}

TEST(Seq2SeqTest, Seq2SeqViterbi) {
  int nclass = 40;
  int hiddendim = 256;
  int inputsteps = 200;
  int maxoutputlen = 100;

  fl::setSeed(1);
  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 1 /* eos token index */,
      nclass - 2 /* pad token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()});

  seq2seq.eval();
  auto input = fl::randn({hiddendim, inputsteps, 1}, fl::dtype::f32);

  auto path = seq2seq.viterbiPath(input);
  ASSERT_GT(path.elements(), 0);
  ASSERT_LE(path.elements(), maxoutputlen);
}

TEST(Seq2SeqTest, Seq2SeqBeamSearchViterbi) {
  int nclass = 40;
  int hiddendim = 256;
  int inputsteps = 200;
  int maxoutputlen = 100;

  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 2 /* eos token index */,
      nclass - 1 /* pad token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()});

  seq2seq.eval();
  auto input = fl::randn({hiddendim, inputsteps, 1}, fl::dtype::f32);

  auto viterbipath = seq2seq.viterbiPath(input);
  auto beampath = seq2seq.beamPath(input, Tensor(), 1);
  ASSERT_EQ(beampath.size(), viterbipath.elements());
  for (int idx = 0; idx < beampath.size(); idx++) {
    ASSERT_EQ(beampath[idx], viterbipath(idx).scalar<int>());
  }
}

TEST(Seq2SeqTest, Seq2SeqMedianWindow) {
  int nclass = 40;
  int hiddendim = 256;
  int inputsteps = 200;
  int maxoutputlen = 100;

  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 2 /* eos token index */,
      nclass - 1 /* pad token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      std::make_shared<MedianWindow>(10, 10));

  seq2seq.eval();
  auto input = fl::randn({hiddendim, inputsteps, 1}, fl::dtype::f32);

  auto viterbipath = seq2seq.viterbiPath(input);
  auto beampath = seq2seq.beamPath(input, Tensor(), 1);
  ASSERT_EQ(beampath.size(), viterbipath.elements());
  for (int idx = 0; idx < beampath.size(); idx++) {
    ASSERT_EQ(beampath[idx], viterbipath(idx).scalar<int>());
  }
}

TEST(Seq2SeqTest, Seq2SeqStepWindow) {
  int nclass = 40;
  int hiddendim = 256;
  int inputsteps = 200;
  int maxoutputlen = 100;

  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 2 /* eos token index */,
      nclass - 1 /* pad token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      std::make_shared<StepWindow>(1, 20, 2.2, 5.8));

  seq2seq.eval();
  auto input = fl::randn({hiddendim, inputsteps, 1}, fl::dtype::f32);

  auto viterbipath = seq2seq.viterbiPath(input);
  auto beampath = seq2seq.beamPath(input, Tensor(), 1);
  ASSERT_EQ(beampath.size(), viterbipath.elements());
  for (int idx = 0; idx < beampath.size(); idx++) {
    ASSERT_EQ(beampath[idx], viterbipath(idx).scalar<int>());
  }
}

TEST(Seq2SeqTest, Seq2SeqStepWindowVectorized) {
  int nclass = 20;
  int hiddendim = 16;
  int batchsize = 2;
  int inputsteps = 20;
  int outputsteps = 10;
  int maxoutputlen = 20;

  Seq2SeqCriterion seq2seq(
      nclass,
      hiddendim,
      nclass - 2 /* eos token index */,
      nclass - 1 /* pad token index */,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      std::make_shared<StepWindow>(0, 5, 2.2, 5.8),
      true);

  auto input = fl::randn({hiddendim, inputsteps, batchsize}, fl::dtype::f32);
  auto target =
      fl::rand({outputsteps, batchsize}, fl::dtype::f32) * 0.99 * nclass;
  target = target.astype(fl::dtype::s32);

  Variable outputV, attentionV, outputS, attentionS;
  std::tie(outputV, attentionV) = seq2seq.vectorizedDecoder(
      noGrad(input), noGrad(target), Tensor(), Tensor());

  std::tie(outputS, attentionS) =
      seq2seq.decoder(noGrad(input), noGrad(target), Tensor(), Tensor());

  ASSERT_TRUE(allClose(outputV, outputS, 1e-6));
  ASSERT_TRUE(allClose(attentionV, attentionS, 1e-6));
}

TEST(Seq2SeqTest, Seq2SeqAttn) {
  int N = 5, H = 8, B = 1, T = 10, U = 5, maxoutputlen = 100;
  Seq2SeqCriterion seq2seq(
      N,
      H,
      N - 2,
      N - 1,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      std::make_shared<MedianWindow>(2, 3));
  seq2seq.eval();

  auto input = noGrad(fl::randn({H, T, B}, fl::dtype::f32));
  auto target = noGrad(
      (fl::rand({U, B}, fl::dtype::f32) * 0.99 * N).astype(fl::dtype::s32));

  Variable output, attention;
  std::tie(output, attention) =
      seq2seq.decoder(input, target, Tensor(), Tensor());
  // check padding works
  ASSERT_EQ(attention.shape(), Shape({U, T, B}));
}

TEST(Seq2SeqTest, Seq2SeqMixedAttn) {
  int N = 5, H = 8, B = 1, T = 10, U = 5, maxoutputlen = 100, nHead = 2;
  Seq2SeqCriterion seq2seq(
      N,
      H,
      N - 2,
      N - 1,
      maxoutputlen,
      {std::make_shared<ContentAttention>(),
       std::make_shared<MultiHeadContentAttention>(H, nHead)},
      std::make_shared<StepWindow>(1, 20, 2.2, 5.8),
      false,
      100,
      0.0,
      false,
      kRandSampling,
      1.0,
      1,
      2);
  seq2seq.eval();

  auto input = noGrad(fl::randn({H, T, B}, fl::dtype::f32));
  auto target = noGrad(
      (fl::rand({U, B}, fl::dtype::f32) * 0.99 * N).astype(fl::dtype::s32));

  Variable output, attention;
  std::tie(output, attention) =
      seq2seq.decoder(input, target, Tensor(), Tensor());
  ASSERT_EQ(attention.shape(), Shape({U * nHead, T, B}));
}

TEST(Seq2SeqTest, Serialization) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const fs::path path = fs::temp_directory_path() / "test.mdl";

  int N = 5, H = 8, B = 1, T = 10, U = 5, maxoutputlen = 100, nAttnRound = 2;

  std::vector<std::shared_ptr<AttentionBase>> attentions(
      nAttnRound, std::make_shared<ContentAttention>());

  auto seq2seq = std::make_shared<Seq2SeqCriterion>(
      N,
      H,
      N - 2,
      N - 1,
      maxoutputlen,
      attentions,
      std::make_shared<MedianWindow>(2, 3),
      false,
      100,
      0.0,
      false,
      kRandSampling,
      1.0,
      2, // nRnnLayer
      nAttnRound,
      0.0);
  seq2seq->eval();

  auto input = noGrad(fl::randn({H, T, B}, fl::dtype::f32));
  auto target = noGrad(
      (fl::rand({U, B}, fl::dtype::f32) * 0.99 * N).astype(fl::dtype::s32));

  Variable output, attention;
  std::tie(output, attention) =
      seq2seq->decoder(input, target, Tensor(), Tensor());

  save(path, seq2seq);

  std::shared_ptr<Seq2SeqCriterion> loaded;
  load(path, loaded);
  loaded->eval();

  Variable outputl, attentionl;
  std::tie(outputl, attentionl) =
      loaded->decoder(input, target, Tensor(), Tensor());

  ASSERT_TRUE(allParamsClose(*loaded, *seq2seq));
  ASSERT_TRUE(allClose(outputl, output));
  ASSERT_TRUE(allClose(attentionl, attention));
}

TEST(Seq2SeqTest, BatchedDecoderStep) {
  int N = 5, H = 8, B = 10, T = 20, maxoutputlen = 100;
  int nRnnLayer = 2, nAttnRound = 2;
  std::vector<std::shared_ptr<AttentionBase>> contentAttentions(
      nAttnRound, std::make_shared<ContentAttention>());
  std::vector<std::shared_ptr<AttentionBase>> neuralContentAttentions(
      nAttnRound, std::make_shared<NeuralContentAttention>(H));

  std::vector<Seq2SeqCriterion> criterions{
      Seq2SeqCriterion(
          N,
          H,
          N - 2,
          N - 1,
          maxoutputlen,
          contentAttentions,
          nullptr,
          false,
          100,
          0.0,
          false,
          kRandSampling,
          1.0,
          nRnnLayer,
          nAttnRound,
          0.0),
      Seq2SeqCriterion(
          N,
          H,
          N - 2,
          N - 1,
          maxoutputlen,
          neuralContentAttentions,
          nullptr,
          false,
          100,
          0.0,
          false,
          kRandSampling,
          1.0,
          nRnnLayer,
          nAttnRound,
          0.0)};

  for (auto& seq2seq : criterions) {
    seq2seq.eval();
    std::vector<Variable> ys;
    std::vector<Seq2SeqState> inStates(B, Seq2SeqState(nAttnRound));
    std::vector<Seq2SeqState*> inStatePtrs(B);

    auto input = noGrad(fl::randn({H, T, 1}, fl::dtype::f32));
    std::vector<std::vector<float>> single_scores(B);
    std::vector<std::vector<float>> batched_scores;

    for (int i = 0; i < B; i++) {
      Variable y = constant(i % N, {1}, fl::dtype::s32, false);
      ys.push_back(y);

      inStates[i].alpha = noGrad(fl::randn({1, T, 1}, fl::dtype::f32));
      for (int j = 0; j < nAttnRound; j++) {
        inStates[i].hidden[j] =
            noGrad(fl::randn({H, 1, nRnnLayer}, fl::dtype::f32));
      }
      inStates[i].summary = noGrad(fl::randn({H, 1, 1}, fl::dtype::f32));
      inStatePtrs[i] = &inStates[i];

      // Single forward
      Seq2SeqState outstate(nAttnRound);
      Variable ox;
      std::tie(ox, outstate) = seq2seq.decodeStep(
          input, y, inStates[i], Tensor(), Tensor(), input.dim(1));
      ox = logSoftmax(ox, 0);
      single_scores[i] = ox.tensor().toHostVector<float>();
    }

    // Batched forward
    std::vector<Seq2SeqStatePtr> outstates;
    std::tie(batched_scores, outstates) =
        seq2seq.decodeBatchStep(input, ys, inStatePtrs);

    // Check
    for (int i = 0; i < B; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_NEAR(single_scores[i][j], batched_scores[i][j], 1e-5);
      }
    }
  }
}

TEST(Seq2SeqTest, Seq2SeqSampling) {
  int N = 5, H = 8, B = 1, T = 10, U = 5, maxoutputlen = 100;
  auto input = noGrad(fl::randn({H, T, B}, fl::dtype::f32));
  auto target = noGrad(
      (fl::rand({U, B}, fl::dtype::f32) * 0.99 * N).astype(fl::dtype::s32));

  std::vector<std::string> samplingStrategy({kRandSampling, kModelSampling});

  for (const auto& ss : samplingStrategy) {
    Seq2SeqCriterion seq2seq(
        N,
        H,
        N - 2,
        N - 1,
        maxoutputlen,
        {std::make_shared<ContentAttention>()},
        nullptr,
        false,
        0,
        0.05,
        false,
        ss);
    seq2seq.train();

    Variable output, attention;
    std::tie(output, attention) =
        seq2seq.decoder(input, target, Tensor(), Tensor());
    ASSERT_EQ(attention.shape(), Shape({U, T, B}));
    ASSERT_EQ(output.shape(), Shape({N, U, B}));
  }

  Seq2SeqCriterion seq2seq1(
      N,
      H,
      N - 2,
      N - 1,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      nullptr,
      false,
      60,
      0.05,
      false,
      kRandSampling);
  seq2seq1.train();

  Variable output, attention;
  std::tie(output, attention) =
      seq2seq1.vectorizedDecoder(input, target, Tensor(), Tensor());
  ASSERT_EQ(attention.shape(), Shape({U, T, B}));
  ASSERT_EQ(output.shape(), Shape({N, U, B}));

  Seq2SeqCriterion seq2seq2(
      N,
      H,
      N - 2,
      N - 1,
      maxoutputlen,
      {std::make_shared<ContentAttention>()},
      nullptr,
      false,
      60,
      0.05,
      false,
      kModelSampling);
  seq2seq2.train();
  ASSERT_THROW(
      seq2seq2.vectorizedDecoder(input, target, Tensor(), Tensor()),
      std::logic_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
