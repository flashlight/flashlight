/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/pkg/runtime/common/SequentialBuilder.h"

using namespace fl;
using namespace fl::pkg::runtime;
using namespace fl::lib;

namespace {

std::string archDir = "";

} // namespace

TEST(SequentialBuilderTest, SeqModule) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "Bidirectional RNN not supported";
  }
  const std::string archfile = pathsConcat(archDir, "arch.txt");
  int nchannel = 4;
  int nclass = 40;
  int batchsize = 2;
  int inputsteps = 100;

  auto model = buildSequentialModule(archfile, nchannel, nclass);

  auto input = fl::randn({inputsteps, 1, nchannel, batchsize}, fl::dtype::f32);

  auto output = model->forward(noGrad(input));

  ASSERT_EQ(output.dims(), Shape({nclass, inputsteps, batchsize}));

  batchsize = 1;
  input = fl::randn({inputsteps, 1, nchannel, batchsize}, fl::dtype::f32);
  output = model->forward(noGrad(input));
  ASSERT_EQ(output.dims(), Shape({nclass, inputsteps, batchsize}));
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
  const std::string path = fl::lib::getTmpPath("test.mdl");
  const std::string archfile = pathsConcat(archDir, "arch.txt");

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
