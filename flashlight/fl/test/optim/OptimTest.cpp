/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "flashlight/fl/common/common.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

TEST(OptimTest, GradNorm) {
  std::vector<Variable> parameters;
  for (int i = 0; i < 5; i++) {
    auto v = Variable(fl::randn({10, 10, 10}), true);
    v = v.astype(fl::dtype::f64);
    v.addGrad(Variable(fl::randn({10, 10, 10}, fl::dtype::f64), false));
    parameters.push_back(v);
  }
  double max_norm = 5.0;
  clipGradNorm(parameters, max_norm);

  double clipped = 0.0;
  for (auto& v : parameters) {
    auto& g = v.grad().tensor();
    clipped += fl::sum(g * g).asScalar<double>();
  }
  clipped = std::sqrt(clipped);
  ASSERT_TRUE(allClose(fl::full({1}, max_norm), fl::full({1}, clipped)));
}

TEST(OptimTest, GradNormF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<Variable> parameters;
  for (int i = 0; i < 5; i++) {
    auto v = Variable(fl::randn({10, 10, 10}), true);
    v = v.astype(fl::dtype::f16);
    v.addGrad(Variable(fl::randn({10, 10, 10}, fl::dtype::f16), false));
    parameters.push_back(v);
  }
  double max_norm = 5.0;
  clipGradNorm(parameters, max_norm);

  double clipped = 0.0;
  for (auto& v : parameters) {
    auto& g = v.grad().tensor();
    clipped += fl::sum(g * g).asScalar<double>();
  }
  clipped = std::sqrt(clipped);
  ASSERT_TRUE(allClose(fl::full({1}, max_norm), fl::full({1}, clipped), 1e-2));
}

TEST(SerializationTest, OptimizerSerialize) {
  const fs::path path = fs::temp_directory_path() / "optmizer.bin";

  std::vector<Variable> parameters;
  for (int i = 0; i < 5; i++) {
    auto v = Variable(fl::randn({10, 10, 10}, fl::dtype::f64), true);
    v.addGrad(Variable(fl::randn({10, 10, 10}, fl::dtype::f64), false));
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
    parameters2[i].addGrad(Variable(parameters[i].grad().tensor(), false));
  }

  opt->step();
  opt2->step();

  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(allClose(parameters[i].tensor(), parameters2[i].tensor()));
  }

  opt = std::make_shared<NovogradOptimizer>(parameters, 0.01);
  opt->step();

  save(
      path, parameters, static_cast<std::shared_ptr<FirstOrderOptimizer>>(opt));
  load(path, parameters2, opt2);

  for (int i = 0; i < 5; i++) {
    parameters2[i].addGrad(Variable(parameters[i].grad().tensor(), false));
  }

  opt->step();
  opt2->step();

  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(allClose(parameters[i].tensor(), parameters2[i].tensor()));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
