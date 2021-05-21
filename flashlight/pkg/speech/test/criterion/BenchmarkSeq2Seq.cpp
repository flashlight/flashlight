/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/flashlight.h"

#include <iomanip>
#include <iostream>

#include <arrayfire.h>
#include <array>

#include "flashlight/pkg/speech/criterion/attention/attention.h"
#include "flashlight/pkg/speech/criterion/criterion.h"

using namespace fl;
using namespace fl::app::asr;

void timeBeamSearch() {
  int N = 40, H = 256, T = 200;

  Seq2SeqCriterion seq2seq(
      // Make eos -1 so beam search runs to outputlen
      N, /* nClass */
      H, /* hiddenDim */
      -1, /* eosIdx */
      N - 1, /* padIdx */
      200, /* maxDecoderOutputLen */
      {std::make_shared<ContentAttention>() /* attentions */});

  auto input = af::randn(H, T, 1, f32);

  // Warmup
  seq2seq.beamPath(input, af::array());

  int iters = 10;
  std::vector<int> beamsizes = {1, 5, 10, 20};
  for (auto b : beamsizes) {
    auto s = af::timer::start();
    for (int i = 0; i < iters; ++i) {
      seq2seq.beamPath(input, af::array(), b);
    }
    fl::sync();
    auto e = af::timer::stop(s);
    std::cout << "Total time (beam size: " << b << ") " << std::setprecision(5)
              << e * 1000.0 / iters << " msec" << std::endl;
  }
}

void timeForwardBackward() {
  int N = 40, H = 256, B = 2, T = 200, U = 50;

  Seq2SeqCriterion seq2seq(
      N, /* nClass */
      H, /* hiddenDim */
      N - 2, /* eosIdx */
      N - 1, /* padIdx */
      0, /* maxDecoderOutputLen */
      {std::make_shared<ContentAttention>()} /* attentions */);

  auto input = Variable(af::randn(H, T, B, f32), true);
  auto target = noGrad((af::randu(U, B, f32) * 0.99 * N).as(s32));

  // Warmup
  for (int i = 0; i < 10; ++i) {
    auto loss = seq2seq({input, target}).front();
    loss.backward();
  }
  fl::sync();

  int iters = 100;
  auto s = af::timer::start();
  for (int i = 0; i < iters; ++i) {
    auto loss = seq2seq({input, target}).front();
    loss.backward();
  }
  fl::sync();
  auto e = af::timer::stop(s);
  std::cout << "Total time (fwd+bwd pass) " << std::setprecision(5)
            << e * 1000.0 / iters << " msec" << std::endl;
}

int main() {
  af::info();
  fl::init();

  timeForwardBackward();
  timeBeamSearch();
  return 0;
}
