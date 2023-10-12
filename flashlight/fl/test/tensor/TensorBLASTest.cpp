/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace ::testing;
using namespace fl;

TEST(TensorBLASTest, matmul) {
  // TODO: test tensors with order > 2

  // Reference impl
  auto matmulRef = [](const Tensor& lhs, const Tensor& rhs) {
    // (M x N) x (N x K) --> (M x K)
    int M = lhs.dim(0);
    int N = lhs.dim(1);
    int K = rhs.dim(1);

    auto out = fl::full({M, K}, 0.);

    for (unsigned i = 0; i < M; ++i) {
      for (unsigned j = 0; j < K; ++j) {
        for (unsigned k = 0; k < N; ++k) {
          out(i, j) += lhs(i, k) * rhs(k, j);
        }
      }
    }
    return out;
  };

  int i = 10;
  int j = 20;
  int k = 12;

  auto a = fl::rand({i, j});
  auto b = fl::rand({j, k});
  auto ref = matmulRef(a, b);
  ASSERT_TRUE(allClose(fl::matmul(a, b), ref));
  ASSERT_TRUE(allClose(
      fl::matmul(
          a,
          fl::transpose(b),
          fl::MatrixProperty::None,
          fl::MatrixProperty::Transpose),
      ref));
  ASSERT_TRUE(allClose(
      fl::matmul(fl::transpose(a), b, fl::MatrixProperty::Transpose), ref));
}

TEST(TensorBLASTest, matmulShapes) {
  using T = fl::MatrixProperty;
  // Matrix/vector/scalar multiplies
  ASSERT_EQ(fl::matmul(fl::rand({10}), fl::rand({10})).shape(), Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose, T::Transpose)
          .shape(),
      Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::None, T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({1, 10}), fl::rand({10})).shape(), Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({1}), fl::rand({1, 10})).shape(), Shape({10}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({3, 4}), fl::rand({4})).shape(), Shape({3}));
  ASSERT_EQ(fl::matmul(fl::rand({5}), fl::rand({5, 7})).shape(), Shape({7}));
  ASSERT_THROW(fl::matmul(fl::rand({1}), fl::rand({10})), std::exception);
  ASSERT_THROW(fl::matmul(fl::rand({3}), fl::rand({5, 7})), std::exception);

  // Batch matrix multiply
  unsigned M = 10;
  unsigned K = 12;
  unsigned N = 14;
  unsigned b2 = 2;
  unsigned b3 = 4;
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({K, N})).shape(), Shape({M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K, b2}), fl::rand({K, N, b2})).shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K, b2, b3}), fl::rand({K, N, b2, b3})).shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K, b2, b3}), fl::rand({K, N})).shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({K, N, b2, b3})).shape(),
      Shape({M, N, b2, b3}));
  // Batch matrix multiply with transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({K, N}), T::Transpose).shape(),
      Shape({M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({M, N}));
  // b2 transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M, b2}), fl::rand({K, N}), T::Transpose).shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K, b2}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({K, N, b2}), T::Transpose).shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({N, K, b2}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M, b2}), fl::rand({K, N, b2}), T::Transpose)
          .shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K, b2}), fl::rand({N, K, b2}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2}));
  // b2, b3 transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M, b2, b3}), fl::rand({K, N}), T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K, b2, b3}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({K, N, b2, b3}), T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K}), fl::rand({N, K, b2, b3}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({K, M, b2, b3}), fl::rand({K, N, b2, b3}), T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K, b2, b3}),
          fl::rand({N, K, b2, b3}),
          T::None,
          T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));

  ASSERT_EQ(
      fl::matmul(
          fl::rand({256, 200, 2}),
          fl::rand({256, 200, 2}),
          T::None,
          T::Transpose)
          .shape(),
      Shape({256, 256, 2}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
