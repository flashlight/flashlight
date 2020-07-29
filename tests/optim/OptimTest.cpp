/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/common/common.h"
#include "flashlight/optim/optim.h"

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
  ASSERT_TRUE(allClose(af::constant(max_norm, 1), af::constant(clipped, 1)));
}

TEST(SerializationTest, OptimizerSerialize) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const std::string path = "/tmp/" + userstr + "_optm";

  std::vector<Variable> parameters;
  for (int i = 0; i < 5; i++) {
    auto v = Variable(af::randn(10, 10, 10, f64), true);
    v.addGrad(Variable(af::randn(10, 10, 10, f64), false));
    parameters.push_back(v);
  }

  std::shared_ptr<FirstOrderOptimizer> opt;
  opt = std::make_shared<AdamOptimizer>(parameters, 0.0001);
  opt->step();

  save(
      path, parameters, static_cast<std::shared_ptr<FirstOrderOptimizer>>(opt));

  std::vector<Variable> parameters2;
  std::shared_ptr<FirstOrderOptimizer> opt2;
  load(path, parameters2, opt2);

  for (int i = 0; i < 5; i++) {
    parameters2[i].addGrad(Variable(parameters[i].grad().array(), false));
  }

  opt->step();
  opt2->step();

  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(allClose(parameters[i].array(), parameters2[i].array()));
  }

  opt = std::make_shared<NovogradOptimizer>(parameters, 0.01);
  opt->step();

  save(
      path, parameters, static_cast<std::shared_ptr<FirstOrderOptimizer>>(opt));
  load(path, parameters2, opt2);

  for (int i = 0; i < 5; i++) {
    parameters2[i].addGrad(Variable(parameters[i].grad().array(), false));
  }

  opt->step();
  opt2->step();

  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(allClose(parameters[i].array(), parameters2[i].array()));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
