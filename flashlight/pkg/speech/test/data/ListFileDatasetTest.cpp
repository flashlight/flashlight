/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/lib/text/String.h"
#include "flashlight/pkg/speech/data/ListFileDataset.h"

using namespace fl::lib;
using namespace fl::pkg::speech;

namespace {
using namespace fl;

fs::path loadPath = "";

auto letterToTarget = [](void* data, Shape dims, fl::dtype /* unused */) {
  std::string transcript(
      static_cast<char*>(data), static_cast<char*>(data) + dims.elements());
  std::vector<int> tgt;
  for (auto c : transcript) {
    tgt.push_back(static_cast<int>(c));
  }
  return Tensor::fromVector(tgt);
};
} // namespace

TEST(ListFileDatasetTest, LoadData) {
  const fs::path dataPath = loadPath / "data.lst";
  if (!fs::exists(dataPath)) {
    throw std::runtime_error(
        "ListFileDatasetTest, LoadData - can't open test data.lst");
  }
  std::vector<std::string> data;
  {
    std::ifstream in(dataPath);
    for (std::string s; std::getline(in, s);) {
      data.emplace_back(s);
    }
  }

  const fs::path rootPath = fs::temp_directory_path() / "data.lst";
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
    ASSERT_EQ(audiods.get(i)[0].shape(), Shape({1, 24000}));
    ASSERT_EQ(audiods.get(i)[1].elements(), expectedTgtLen[i]);
    ASSERT_EQ(audiods.get(i)[1].elements(), audiods.getTargetSize(i));
    ASSERT_TRUE(audiods.get(i)[2].isEmpty());
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
  loadPath = fs::path(DATA_TEST_DATADIR);
#endif

  return RUN_ALL_TESTS();
}
