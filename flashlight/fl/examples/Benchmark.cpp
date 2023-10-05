/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>

#include "flashlight/fl/common/Timer.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/tensor.h"

using namespace fl;

#define TIME(FUNC)                                           \
  std::cout << "Timing " << #FUNC << " ...  " << std::flush; \
  std::cout << std::setprecision(5) << FUNC() * 1000.0 << " msec" << std::endl;

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 10; ++i) {
    fn();
  }
  fl::sync();

  int num_iters = 100;
  fl::sync();
  auto start = fl::Timer::start();
  for (int i = 0; i < num_iters; i++) {
    fn();
  }
  fl::sync();
  return fl::Timer::stop(start) / num_iters;
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

  auto input = Variable(fl::rand({224, 224, 3, 128}) * 2 - 2, false);

  auto b = model.forward(input);
  auto gradoutput = Variable(fl::rand(b.shape()) * 2 - 2, false);

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
  Variable input(
      (fl::rand({num_elems}) * vocab_size).astype(fl::dtype::s32), false);
  Variable grad_output(
      fl::randn({embed_dim, num_elems}, fl::dtype::f32), false);

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
  Variable input(fl::rand({N, T, B}, fl::dtype::f32), true);
  Variable dout(fl::rand({M, T, B}, fl::dtype::f32), false);
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
  Variable input(fl::rand({W, H, C, N}, fl::dtype::f32), true);
  Variable dout(fl::rand({W, H, C, N}, fl::dtype::f32), true);
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
  Variable input(fl::rand({W, H, C, N}, fl::dtype::f32), true);
  Variable dout(fl::rand({W, H, C, N}, fl::dtype::f32), true);
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
  fl::init();
  TIME(alexnet);
  TIME(embedding);
  TIME(linear);
  TIME(batchNorm);
  TIME(layerNorm);
  return 0;
}
