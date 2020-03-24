/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>

#include "flashlight/nn/nn.h"

using namespace fl;

#define TIME(FUNC)                                           \
  std::cout << "Timing " << #FUNC << " ...  " << std::flush; \
  std::cout << std::setprecision(5) << FUNC() * 1000.0 << " msec" << std::endl;

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 10; ++i) {
    fn();
  }
  af::sync();

  int num_iters = 100;
  af::sync();
  auto start = af::timer::start();
  for (int i = 0; i < num_iters; i++) {
    fn();
  }
  af::sync();
  return af::timer::stop(start) / num_iters;
}

double alexnet() {
  Sequential model;
  model.add(Conv2D(3, 64, 11, 11, 4, 4, 2, 2)); // 224 -> 55
  model.add(ReLU());
  model.add(Pool2D(3, 3, 2, 2)); // 55 ->  27
  model.add(Conv2D(64, 192, 5, 5, 1, 1, 2, 2)); //  27 -> 27
  model.add(ReLU());
  model.add(Pool2D(3, 3, 2, 2)); //  27 ->  13
  model.add(Conv2D(192, 384, 3, 3, 1, 1, 1, 1)); //  13 ->  13
  model.add(ReLU());
  model.add(Conv2D(384, 256, 3, 3, 1, 1, 1, 1)); //  13 ->  13
  model.add(ReLU());
  model.add(Conv2D(256, 256, 3, 3, 1, 1, 1, 1)); //  13 ->  13
  model.add(ReLU());
  model.add(Pool2D(3, 3, 2, 2)); // 13 -> 6

  auto input = Variable(af::randu(224, 224, 3, 128) * 2 - 2, false);

  auto b = model.forward(input);
  auto gradoutput = Variable(af::randu(b.dims()) * 2 - 2, false);

  auto alexnet_fn = [&]() {
    auto output = model.forward(input);
    output.backward(gradoutput);
  };
  return timeit(alexnet_fn);
}

double embedding() {
  int embed_dim = 256;
  int vocab_size = 10000;

  Embedding embed(embed_dim, vocab_size);

  int num_elems = 400;
  Variable input((af::randu(num_elems) * vocab_size).as(s32), false);
  Variable grad_output(af::randn(embed_dim, num_elems, f32), false);

  auto embed_fn = [&]() {
    embed.zeroGrad();
    auto output = embed(input);
    output.backward(grad_output);
  };
  return timeit(embed_fn);
}

double linear() {
  int M = 256;
  int N = 512;
  int B = 8;
  int T = 2;
  Variable input(af::randu(N, T, B, f32), true);
  Variable dout(af::randu(M, T, B, f32), false);
  Linear lin(N, M);

  auto lin_fn = [&]() {
    lin.zeroGrad();
    input.zeroGrad();
    auto output = lin(input);
    output.backward(dout);
  };

  return timeit(lin_fn);
}

double batchNorm() {
  // Takes around 0.72 ms on Tesla M40 with cudnn torch
  int N = 8;
  int C = 512;
  int H = 32;
  int W = 32;
  Variable input(af::randu(W, H, C, N, f32), true);
  Variable dout(af::randu(W, H, C, N, f32), true);
  BatchNorm bn(2, C); // Spatial batchnorm

  auto bn_fn = [&]() {
    bn.zeroGrad();
    input.zeroGrad();
    auto output = bn(input);
    output.backward(dout);
  };

  return timeit(bn_fn);
}

double layerNorm() {
  // Takes around 7.8 ms on Tesla M40 with cudnn torch
  int N = 8;
  int C = 512;
  int H = 32;
  int W = 32;
  Variable input(af::randu(W, H, C, N, f32), true);
  Variable dout(af::randu(W, H, C, N, f32), true);
  LayerNorm ln(3);

  auto ln_fn = [&]() {
    ln.zeroGrad();
    input.zeroGrad();
    auto output = ln(input);
    output.backward(dout);
  };

  return timeit(ln_fn);
}

int main() {
  af::info();
  TIME(alexnet);
  TIME(embedding);
  TIME(linear);
  TIME(batchNorm);
  TIME(layerNorm);
  return 0;
}
