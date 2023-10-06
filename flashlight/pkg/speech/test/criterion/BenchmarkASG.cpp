/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/flashlight.h"

#include <iomanip>
#include <iostream>

#include <array>

#include "flashlight/pkg/speech/criterion/criterion.h"

using namespace fl;
using namespace fl::pkg::speech;

int main() {
  fl::setDevice(0);
  fl::init();

  int N = 30, T = 487, L = 34, B = 20;

  auto asg = AutoSegmentationCriterion(N);

  auto input = Variable(fl::rand({N, T, B}) * 2 - 1, true);

  auto target = Variable(
      fl::abs(fl::rand({L, B}, fl::dtype::s32)).astype(fl::dtype::s32) %
          (N - 1),
      false);

  int ntimes = 50;
  Variable b = asg.forward({input, target}).front();
  Variable gradoutput = Variable(fl::rand(b.shape()) * 2 - 2, false);
  for (int i = 0; i < 5; ++i) {
    b = asg.forward({input, target}).front();
    b.backward();
  }
  fl::sync();
  auto s = fl::Timer::start();
  for (int i = 0; i < ntimes; ++i) {
    b = asg.forward({input, target}).front();
    b.backward(gradoutput);
  }
  fl::sync();
  auto e = fl::Timer::stop(s);
  std::cout << "Total time (fwd+bwd pass) " << std::setprecision(5)
            << e * 1000.0 / ntimes << " msec" << std::endl;
  return 0;
}
