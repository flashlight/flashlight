/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdint.h>
#include <unordered_map>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/fl/flashlight.h"

#include "flashlight/app/asr/runtime/runtime.h"
#include "flashlight/ext/common/Serializer.h"

#include "flashlight/lib/common/System.h"

using namespace fl::app::asr;

namespace {
const std::string kPath = fl::lib::getTmpPath("test.mdl");

bool afEqual(const fl::Variable& a, const fl::Variable& b) {
  if (a.isCalcGrad() != b.isCalcGrad()) {
    return false;
  }
  if (a.dims() != b.dims()) {
    return false;
  }
  if (a.array().isempty() && b.array().isempty()) {
    return true;
  }
  return af::allTrue<bool>(af::abs(a.array() - b.array()) < 1E-7);
}

} // namespace

TEST(RuntimeTest, LoadAndSave) {
  std::unordered_map<std::string, std::string> config(
      {{"date", "01-01-01"}, {"lr", "0.1"}, {"user", "guy_fawkes"}});
  fl::Sequential model;
  model.add(fl::Conv2D(4, 6, 2, 1));
  model.add(fl::GatedLinearUnit(2));
  model.add(fl::Dropout(0.2));
  model.add(fl::Conv2D(3, 4, 3, 1, 1, 1, 0, 0, 1, 1, false));
  model.add(fl::GatedLinearUnit(2));
  model.add(fl::Dropout(0.214));

  fl::ext::Serializer::save(kPath, FL_APP_ASR_VERSION, config, model);

  fl::Sequential modelload;
  std::unordered_map<std::string, std::string> configload;
  std::string versionload;
  fl::ext::Serializer::load(kPath, versionload, configload, modelload);

  EXPECT_EQ(configload.size(), config.size());
  EXPECT_EQ(versionload, FL_APP_ASR_VERSION);
  EXPECT_THAT(config, ::testing::ContainerEq(configload));

  ASSERT_EQ(model.prettyString(), modelload.prettyString());

  model.eval();
  modelload.eval();

  for (int i = 0; i < 10; ++i) {
    auto in = fl::Variable(af::randu(10, 1, 4), i & 1);
    ASSERT_TRUE(afEqual(model.forward(in), modelload.forward(in)));
  }
}

TEST(RuntimeTest, TestCleanFilepath) {
  auto s = cleanFilepath("timit/train.\\mymodel");
#ifdef _WIN32
  ASSERT_EQ(s, "timit/train.#mymodel");
#else
  ASSERT_EQ(s, "timit#train.\\mymodel");
#endif
}

TEST(RuntimeTest, SpeechStatMeter) {
  SpeechStatMeter meter;
  std::array<int, 5> a{1, 2, 3, 4, 5};
  std::array<int, 6> b{1, 1, 3, 3, 5, 6};
  meter.add(af::array(5, a.data()), af::array(6, b.data()));
  af::array out;
  auto stats1 = meter.value();
  ASSERT_EQ(stats1[0], 5.0);
  ASSERT_EQ(stats1[1], 6.0);
  ASSERT_EQ(stats1[2], 5.0);
  ASSERT_EQ(stats1[3], 6.0);
  ASSERT_EQ(stats1[4], 1.0);
  meter.add(af::array(3, a.data() + 1), af::array(3, b.data()));
  auto stats2 = meter.value();
  ASSERT_EQ(stats2[0], 8.0);
  ASSERT_EQ(stats2[1], 9.0);
  ASSERT_EQ(stats2[2], 5.0);
  ASSERT_EQ(stats2[3], 6.0);
  ASSERT_EQ(stats2[4], 2.0);
}

TEST(RuntimeTest, parseValidSets) {
  std::string in = "";
  auto op = parseValidSets(in);
  ASSERT_EQ(op.size(), 0);

  std::string in1 = "d1:d1.lst,d2:d2.lst";
  auto op1 = parseValidSets(in1);
  ASSERT_EQ(op1.size(), 2);
  ASSERT_EQ(op1[0], (std::pair<std::string, std::string>("d1", "d1.lst")));
  ASSERT_EQ(op1[1], (std::pair<std::string, std::string>("d2", "d2.lst")));

  std::string in2 = "d1.lst,d2.lst,d3.lst";
  auto op2 = parseValidSets(in2);
  ASSERT_EQ(op2.size(), 3);
  ASSERT_EQ(op2[0], (std::pair<std::string, std::string>("d1.lst", "d1.lst")));
  ASSERT_EQ(op2[1], (std::pair<std::string, std::string>("d2.lst", "d2.lst")));
  ASSERT_EQ(op2[2], (std::pair<std::string, std::string>("d3.lst", "d3.lst")));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
