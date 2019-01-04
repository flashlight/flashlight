/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/optim/optim.h"
#include "flashlight/common/common.h"

using namespace fl;

TEST(OptimTest, GradNorm) {
  std::vector<Variable> parameters;
  for (int i = 0; i < 5; i++) {
    auto v = Variable(af::array(), true);
    v.addGrad(Variable(af::randn(10, 10, 10, f64), false));
    parameters.push_back(v);
  }
  double max_norm = 5.0;
  clipGradNorm(parameters, max_norm);

  double clipped = 0.0;
  for (auto& v : parameters) {
    auto& g = v.grad().array();
    clipped += af::sum<double>(g * g);
  }
  clipped = std::sqrt(clipped);
  ASSERT_TRUE(allClose(max_norm, clipped));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
