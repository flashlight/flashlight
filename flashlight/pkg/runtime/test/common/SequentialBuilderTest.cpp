/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/pkg/runtime/common/SequentialBuilder.h"

using namespace fl;
using namespace fl::pkg::runtime;

namespace {

fs::path archDir = "";

} // namespace

TEST(SequentialBuilderTest, SeqModule) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "Bidirectional RNN not supported";
  }
  const fs::path archfile = archDir / "arch.txt";
  int nchannel = 4;
  int nclass = 40;
  int batchsize = 2;
  int inputsteps = 100;

  auto model = buildSequentialModule(archfile, nchannel, nclass);

  auto input = fl::randn({inputsteps, 1, nchannel, batchsize}, fl::dtype::f32);

  auto output = model->forward(noGrad(input));

  ASSERT_EQ(output.shape(), Shape({nclass, inputsteps, batchsize}));

  batchsize = 1;
  input = fl::randn({inputsteps, 1, nchannel, batchsize}, fl::dtype::f32);
  output = model->forward(noGrad(input));
  ASSERT_EQ(output.shape(), Shape({nclass, inputsteps, batchsize}));
}

TEST(SequentialBuilderTest, Serialization) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "Bidirectional RNN not supported";
  }
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const fs::path path = fs::temp_directory_path() / "test.mdl";
  const fs::path archfile = archDir / "arch.txt";

  int C = 1, N = 5, B = 1, T = 10;
  auto model = buildSequentialModule(archfile, C, N);

  auto input = noGrad(fl::randn({T, 1, C, B}, fl::dtype::f32));
  auto output = model->forward(input);

  save(path, model);

  std::shared_ptr<Sequential> loaded;
  load(path, loaded);

  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded.get(), *model));
  ASSERT_TRUE(allClose(outputl.tensor(), output.tensor()));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();

// Resolve directory for arch
#ifdef ARCHDIR
  archDir = ARCHDIR;
#endif

  return RUN_ALL_TESTS();
}
