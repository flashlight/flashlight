/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradReductionTest, Sum) {
  for (const bool keepDims : {false, true}) {
    Shape s = {6};
    if (keepDims) {
      s = {6, 1};
    }

    auto x = Variable(fl::rand(s), true);
    auto y = Variable(fl::rand({6, 3}), true);

    auto z = x * sum(y, {1}, keepDims);
    auto dz = Variable(fl::full(s, 1.0), false);
    z.backward(dz);

    auto dy = y.grad();
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 3})));
    ASSERT_TRUE(allClose(dx.tensor(), fl::sum(y.tensor(), {1}, keepDims)));

    // Reduce over 1-dim input
    auto funcMean_0 = [keepDims](const Variable& in) {
      return sum(in, {0}, keepDims);
    };
    auto in = Variable(fl::rand({6}), true);
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean_0, in, 5E-3));
    // Reduce over scalar input
    auto inScalar = Variable(fl::fromScalar(3.14), true);
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean_0, inScalar, 5E-3));
  }

  auto r = Variable(fl::rand({5, 6, 7, 8}), true);
  auto rOut = sum(r, {1, 2});
  auto rOutTensor = fl::sum(r.tensor(), {1, 2});
  ASSERT_TRUE(allClose(rOut.tensor(), rOutTensor));
}

TEST(AutogradReductionTest, SumAs) {
  auto x = Variable(fl::rand({5}), true);
  auto y = Variable(fl::rand({5, 2}), true);
  auto z = x * sumAs(y, x);
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 2})));
  ASSERT_TRUE(allClose(dx.tensor(), fl::sum(y.tensor(), {1})));
}

TEST(AutogradReductionTest, SumAs2) {
  auto y = Variable(fl::rand({5, 2}), true);
  auto z = sumAs(y, {5});
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dy.tensor(), fl::full({5, 2}, 1.0)));
}

TEST(AutogradReductionTest, Mean) {
  for (const bool keepDims : {false, true}) {
    Shape xShape = keepDims ? Shape({5, 1, 1}) : Shape({5});
    auto x = Variable(fl::rand(xShape), true);
    auto y = Variable(fl::rand({5, 3, 2}), true);
    auto varOut = mean(y, {1, 2}, keepDims);
    auto z = x * mean(y, {1, 2}, keepDims);
    auto dz = Variable(fl::full(x.shape(), 1.0), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 3, 2}) / 6));
    ASSERT_TRUE(allClose(dx.tensor(), fl::mean(y.tensor(), {1, 2}, keepDims)));

    auto a = Variable(fl::rand({5, 3, 2}, fl::dtype::f64), true);
    auto funcMean = [keepDims](Variable& in) {
      return mean(in, {1, 2}, keepDims);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean, a, 1E-4));

    auto q = Variable(fl::rand({5, 6, 7, 8}), false);
    auto qOut = mean(q, {1, 2}, keepDims);
    auto qOutTensor = fl::mean(q.tensor(), {1, 2}, keepDims);
    ASSERT_TRUE(allClose(qOut.tensor(), qOutTensor));

    auto funcMean_0 = [keepDims](Variable& in) {
      return mean(in, {0}, keepDims);
    };
    // Reduce over 1-dim input
    auto in = Variable(fl::rand({6}), true);
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean_0, in, 5E-3));
    // Reduce over scalar input
    auto inScalar = Variable(fl::fromScalar(3.14), true);
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMean_0, inScalar, 5E-3));
  }
}

TEST(AutogradReductionTest, Variance) {
  std::vector<bool> biased = {true, false};
  for (auto b : biased) {
    for (const bool keepDims : {false, true}) {
      auto x = Variable(fl::rand({5, 6, 7, 8}, fl::dtype::f64), true);

      // TODO:{fl::Tensor} -- enforce AF versioning and remediate
      // Behavior of the bias parameter in af::var was changed in
      // https://git.io/Jv5gF and is different in ArrayFire v3.7. If isbiased is
      // true, sample variance rather than population variance is used. The
      // flashlight API implements the opposite behavior to be consistent with
      // other libraries.
      bool afVarBiasArg = !b;

      auto expectedVar = fl::var(x.tensor(), {1}, afVarBiasArg, keepDims);
      auto calculatedVar = var(x, {1}, b, keepDims);
      ASSERT_TRUE(allClose(calculatedVar.tensor(), expectedVar));

      auto funcVar = [b, keepDims](Variable& in) {
        return var(in, {1, 2}, b, keepDims);
      };
      ASSERT_TRUE(fl::detail::jacobianTestImpl(funcVar, x, 1E-5, 1E-5));
    }
  }
}

TEST(AutogradReductionTest, Norm) {
  auto x = Variable(fl::rand({5, 3}, fl::dtype::f64), true);
  for (const bool keepDims : {false, true}) {
    auto funcNorm2 = [keepDims](Variable& in) {
      return norm(in, {1}, 2, keepDims);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcNorm2, x, 1E-4));
    auto funcNorm1 = [keepDims](Variable& in) {
      return norm(in, {1}, 1, keepDims);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcNorm1, x, 1E-4));
    auto funcNorm3 = [keepDims](Variable& in) {
      return norm(in, {1}, 3, keepDims);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcNorm3, x, 1E-4));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
