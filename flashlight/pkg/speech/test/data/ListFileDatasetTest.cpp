/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fstream>
#include <iostream>
#include <string>

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/pkg/speech/data/ListFileDataset.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl::lib;
using namespace fl::app::asr;

namespace {
std::string loadPath = "";
auto letterToTarget = [](void* data, af::dim4 dims, af::dtype /* unused */) {
  std::string transcript(
      static_cast<char*>(data), static_cast<char*>(data) + dims.elements());
  std::vector<int> tgt;
  for (auto c : transcript) {
    tgt.push_back(static_cast<int>(c));
  }
  return af::array(tgt.size(), tgt.data());
};
} // namespace

TEST(ListFileDatasetTest, LoadData) {
  auto data = getFileContent(pathsConcat(loadPath, "data.lst"));
  const std::string rootPath = fl::lib::getTmpPath("data.lst");
  std::ofstream out(rootPath);
  for (auto& d : data) {
    replaceAll(d, "<TESTDIR>", loadPath);
    out << d;
    out << "\n";
  }
  out.close();
  ListFileDataset audiods(rootPath, nullptr, letterToTarget);
  ASSERT_EQ(audiods.size(), 3);
  std::vector<int> expectedTgtLen = {45, 23, 26};
  std::vector<float> expectedDuration = {1.2, 2.1, 0.6};
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(audiods.get(i).size(), 7);
    ASSERT_EQ(audiods.get(i)[0].dims(), af::dim4(1, 24000));
    ASSERT_EQ(audiods.get(i)[1].elements(), expectedTgtLen[i]);
    ASSERT_EQ(audiods.get(i)[1].elements(), audiods.getTargetSize(i));
    ASSERT_TRUE(audiods.get(i)[2].isempty());
    ASSERT_EQ(audiods.get(i)[3].elements(), 1);
    ASSERT_GE(audiods.get(i)[4].elements(), 15);
    ASSERT_EQ(audiods.get(i)[5].elements(), 1);
    ASSERT_EQ(audiods.get(i)[5].scalar<float>(), expectedDuration[i]);
    ASSERT_EQ(audiods.get(i)[6].elements(), 1);
    ASSERT_EQ(audiods.get(i)[6].scalar<float>(), expectedTgtLen[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
