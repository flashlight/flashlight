/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/fl/tensor/TensorBase.h"

using namespace ::testing;

/*
 * Below this point are ArrayFire-specific implementation tests. They should be
 * [re]moved to a specific ArrayFire backend implementation (along with other
 * flashlight/fl/tensor assets once migration is complete).
 */

namespace {
// TODO:fl::Tensor {testing} copied from fl/common/Utils for testing since
// fl/tensor can't depend on fl/common with Buck. Move this to some other
// location later.
bool allClose(
    const af::array& a,
    const af::array& b,
    double absTolerance = 1e-5) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.dims() != b.dims()) {
    return false;
  }
  if (a.isempty() && b.isempty()) {
    return true;
  }
  return af::max<double>(af::abs(a - b)) < absTolerance;
}
} // namespace

TEST(TensorBaseTest, AfRefCountBasic) {
  // Sanity check that af::arrays moved into fl::Tensors don't have their
  // refcount inrcremented/show proper usage of refs in tensor ops
  int refCount = 0;
  auto a = af::constant(1, {2, 2});
  af_get_data_ref_count(&refCount, a.get());
  ASSERT_EQ(refCount, 1);

  auto tensor = fl::Tensor(std::move(a));
  auto& aRef = tensor.getArray();
  af_get_data_ref_count(&refCount, aRef.get());
  ASSERT_EQ(refCount, 1);
  // Sanity check copying bumps things
  auto aNoRef = tensor.getArray();
  af_get_data_ref_count(&refCount, aNoRef.get());
  ASSERT_EQ(refCount, 2);
}

TEST(TensorBaseTest, BinaryOperators) {
  // TODO:fl::Tensor {testing} expand this test
  // Ensure that some binary operators work properly.
  auto a = fl::Tensor(af::constant(1, {2, 2}));
  auto b = fl::Tensor(af::constant(2, {2, 2}));
  auto c = fl::Tensor(af::constant(3, {2, 2}));

  ASSERT_TRUE(allClose((a == b).getArray(), (a.getArray() == b.getArray())));
  ASSERT_TRUE(allClose((a == b).getArray(), eq(a, b).getArray()));
  ASSERT_TRUE(allClose((a + b).getArray(), c.getArray()));
  ASSERT_TRUE(allClose((a + b).getArray(), add(a, b).getArray()));
}
