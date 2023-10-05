/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradNormalizationTest, Normalize) {
  auto x = Variable(fl::rand({5, 3}, fl::dtype::f64), true);
  auto funcNormalize2 = [](Variable& in) { return normalize(in, {1}); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcNormalize2, x));
  auto ys = funcNormalize2(x);
  ASSERT_TRUE(allClose(
      fl::sum(ys.tensor() * ys.tensor(), {1}),
      fl::full({5}, 1, fl::dtype::f64)));
  auto yb = normalize(x, {1}, 2, 1);
  ASSERT_TRUE(fl::all(fl::sqrt(fl::sum(yb.tensor() * yb.tensor(), {1})) <= 1)
                  .scalar<char>());
}

TEST(AutogradNormalizationTest, BatchNormEvalModeOutputSingleAxis) {
  int featDims = 3;
  std::vector<int> featAxes = {2};
  // input order: HWCN, following the docs
  auto input = Variable(fl::rand({13, 13, featDims, 16}), false);
  auto runningMean = Variable(fl::rand({featDims}, input.type()), false);
  auto runningVar = Variable(fl::rand({featDims}, input.type()), false);
  auto weight = Variable(fl::rand({featDims}, input.type()), false);
  auto bias = Variable(fl::rand({featDims}, input.type()), false);

  auto out = (batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < featDims; ++i) {
    std::array<fl::Index, 4> sel = {fl::span, fl::span, i, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.tensor().flatten()(i).scalar<float>();
    auto thisVar = runningVar.tensor().flatten()(i).scalar<float>();
    auto thisWeight = weight.tensor().flatten()(i).scalar<float>();
    auto thisBias = bias.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1E-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1E-5));
  }

  // test on empty weigts and bias
  out = (batchnorm(
      input,
      Variable(),
      Variable(),
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < featDims; ++i) {
    std::array<fl::Index, 4> sel = {fl::span, fl::span, i, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.tensor().flatten()(i).scalar<float>();
    auto thisVar = runningVar.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1E-5);
    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1E-5));
  }
}

TEST(AutogradNormalizationTest, BatchNormEvalModeOutputMultipleAxis) {
  // input order: HWCN, following the docs
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(fl::rand({13, 13, 4, 16}), false);

  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, input.type()), false);
  auto runningVar = Variable(fl::rand({nfeatures}, input.type()), false);
  auto weight = Variable(fl::rand({nfeatures}, input.type()), false);
  auto bias = Variable(fl::rand({nfeatures}, input.type()), false);

  auto out = (batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < nfeatures; ++i) {
    std::array<fl::Index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.tensor().flatten()(i).scalar<float>();
    auto thisVar = runningVar.tensor().flatten()(i).scalar<float>();
    auto thisWeight = weight.tensor().flatten()(i).scalar<float>();
    auto thisBias = bias.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1e-5);
    expectedOut = expectedOut * thisWeight + thisBias;

    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-4));
  }

  // test on empty weigts and bias
  out = (batchnorm(
      input,
      Variable(),
      Variable(),
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < nfeatures; ++i) {
    std::array<fl::Index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.tensor().flatten()(i).scalar<float>();
    auto thisVar = runningVar.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1e-5);
    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 5e-5));
  }
}

TEST(AutogradNormalizationTest, BatchNormTrainModeOutputSingleAxis) {
  int numFeat = 3;
  std::vector<int> featAxes = {2};
  double epsilon = 1E-5;
  auto input = Variable(fl::rand({13, 13, numFeat, 8}), true);
  auto weight = Variable(fl::rand({numFeat}), true);
  auto bias = Variable(fl::rand({numFeat}), true);
  auto runningMean = Variable(fl::rand({numFeat}), false);
  auto runningVar = Variable(fl::rand({numFeat}), false);

  auto out = batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      true,
      0.0,
      epsilon);

  auto todim = Shape({1, 1, numFeat});
  std::vector<int> nrmAxes = {0, 1, 3};
  auto avg = moddims(mean(input, nrmAxes), todim);
  auto variance =
      moddims(var(input, nrmAxes, true /* population var */), todim);
  auto expectedOut = (input - tileAs(avg, input)) /
      fl::sqrt(tileAs(variance, input) + epsilon);
  expectedOut = expectedOut * tileAs(moddims(weight, todim), input) +
      tileAs(moddims(bias, todim), input);
  ASSERT_TRUE(allClose(out.tensor(), expectedOut.tensor(), 1e-5));
}

TEST(AutogradNormalizationTest, BatchNormTrainModeOutputMultipleAxis) {
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(fl::rand({13, 13, 4, 8}), true);

  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto weight = Variable(fl::rand({nfeatures}), true);
  auto bias = Variable(fl::rand({nfeatures}), true);
  auto runningMean = Variable(fl::rand({nfeatures}), false);
  auto runningVar = Variable(fl::rand({nfeatures}), false);

  auto out = batchnorm(
      input, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);

  auto todim = Shape({nfeatures});
  std::vector<int> nrmAxes = {3};
  auto avg = moddims(mean(input, nrmAxes), todim);
  auto variance = moddims(var(input, nrmAxes, true), todim);

  for (int i = 0; i < nfeatures; ++i) {
    std::array<fl::Index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = avg.tensor().flatten()(i).scalar<float>();
    auto thisVar = variance.tensor().flatten()(i).scalar<float>();
    auto thisWeight = weight.tensor().flatten()(i).scalar<float>();
    auto thisBias = bias.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / std::sqrt(thisVar + 1e-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-5));
  }
}

TEST(AutogradNormalizationTest, BatchNormJacobian) {
  // Jacobian Test with trainMode = true;

  int numFeat = 3;
  std::vector<int> featAxes = {2};
  auto input = Variable(fl::rand({8, 8, numFeat, 16}, fl::dtype::f32), true);
  auto runningMean = Variable(fl::rand({numFeat}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({numFeat}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({numFeat}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({numFeat}, fl::dtype::f32), true);

  auto funcBnIn = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnIn, input, 1e-2, 1e-4));

  auto funcBnWt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnWt, weight, 1e-2, 1e-4));

  auto funcBnBs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnBs, bias, 1e-2, 1e-4));
}

TEST_F(AutogradTestF16, BatchNormJacobianF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Jacobian Test with trainMode = true;

  int numFeat = 3;
  std::vector<int> featAxes = {2};
  auto input = Variable(fl::rand({8, 8, numFeat, 16}, fl::dtype::f16), true);
  auto runningMean = Variable(fl::rand({numFeat}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({numFeat}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({numFeat}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({numFeat}, fl::dtype::f32), true);

  // Use larger perturbations to ensure gradients don't underflow with fp16

  auto funcBnIn = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnIn, input, 5e-2, 1e-1));

  auto funcBnWt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnWt, weight, 5e-2, 1e-1));

  auto funcBnBs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnBs, bias, 5e-2, 1e-1));
}

TEST(AutogradNormalizationTest, BatchNormJacobianMultipleAxes) {
  // Jacobian Test with  trainMode = true;
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(fl::rand({8, 8, 3, 16}, fl::dtype::f32), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcBnIn = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnIn, input, 1e-2, 1e-3));

  auto funcBnWt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnWt, weight, 1e-2, 1e-3));

  auto funcBnBs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnBs, bias, 1e-2, 1e-3));
}

TEST_F(AutogradTestF16, BatchNormJacobianMultipleAxesF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Jacobian Test with trainMode = true;
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(fl::rand({2, 2, 2, 1}, fl::dtype::f16), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);

  // Use larger perturbations to ensure gradients don't underflow with fp16

  auto funcBnIn = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(
      funcBnIn, input, 5e-2, 1e-1)); // TODO: investigate

  auto funcBnWt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnWt, weight, 5e-2, 1e-1));

  auto funcBnBs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcBnBs, bias, 5e-2, 1e-1));
}

TEST(AutogradNormalizationTest, LayerNormJacobian) {
  std::vector<int> featAxes = {0, 1, 2, 3};
  auto input = Variable(fl::rand({7, 7, 3, 10}), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcLnIn = [&](Variable& in) {
    return batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);
  };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLnIn, input, 1e-2, 1e-4));
}

TEST_F(AutogradTestF16, LayerNormJacobianF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<int> featAxes = {0, 1, 2, 3};
  const float inputScale = 4.0; // scale the input to prevent grad underflow
  auto input =
      Variable(inputScale * fl::rand({2, 2, 2, 4}, fl::dtype::f16), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcLnIn = [&](Variable& in) {
    return batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);
  };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLnIn, input, 1e-4, 1e-2));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
