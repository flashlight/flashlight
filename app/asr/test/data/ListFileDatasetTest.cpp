/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fstream>
#include <iostream>
#include <string>

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/app/asr/data/ListFileDataset.h"

#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl;
using namespace fl::lib;
using namespace fl::tasks::asr;

namespace {
std::string loadPath = "";
}

TEST(ListFileDatasetTest, LoadData) {
  auto data = getFileContent(pathsConcat(loadPath, "data.lst"));
  auto rootPath = "/tmp/data.lst";
  std::ofstream out(rootPath);
  for (auto& d : data) {
    replaceAll(d, "<TESTDIR>", loadPath);
    out << d;
    out << "\n";
  }
  out.close();
  ListFileDataset audiods(rootPath);
  ASSERT_EQ(audiods.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(audiods.get(i).size(), 4);
    ASSERT_EQ(audiods.get(i)[0].dims(), af::dim4(1, 24000));
    ASSERT_EQ(audiods.get(i)[3].elements(), 1);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
