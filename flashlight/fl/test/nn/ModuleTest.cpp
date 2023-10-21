/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

namespace {

class ContainerTestClass : public Sequential {
 public:
  void addParam(const Variable& param) {
    params_.push_back(param);
  }
};

} // namespace

TEST(ModuleTest, ContainerReplaceParam) {
  auto seq = ContainerTestClass();
  seq.addParam(Variable(fl::rand({5, 5}), true));
  seq.add(Linear(10, 20));
  seq.addParam(Variable(fl::rand({5, 5}), true));
  seq.add(ReLU());
  seq.add(Linear(20, 30));
  seq.addParam(Variable(fl::rand({5, 5}), true));

  // Change the first parameter
  auto new_param = Variable(fl::rand({5, 5}), true);
  seq.setParams(new_param, 0);
  ASSERT_TRUE(allClose(seq.params()[0], new_param));

  // Change the first linear layer's first parameter
  new_param = Variable(fl::rand({10, 20}), true);
  seq.setParams(new_param, 1);
  ASSERT_TRUE(allClose(seq.params()[1], new_param));
  ASSERT_TRUE(allClose(seq.module(0)->param(0), new_param));

  // Change the second linear layer's first parameter
  new_param = Variable(fl::rand({20, 30}), true);
  seq.setParams(new_param, 4);
  ASSERT_TRUE(allClose(seq.params()[4], new_param));
  ASSERT_TRUE(allClose(seq.module(2)->param(0), new_param));

  // Change the last parameter
  new_param = Variable(fl::rand({5, 5}), true);
  seq.setParams(new_param, 6);
  ASSERT_TRUE(allClose(seq.param(6), new_param));
}
