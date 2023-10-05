/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/flashlight.h"

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/pkg/runtime/common/SequentialBuilder.h"

using namespace fl;
using namespace fl::pkg::runtime;

namespace {

fs::path archDir = "";

} // namespace

TEST(ConvLmModuleTest, GCNN14BAdaptiveSoftmax) {
  const fs::path archfile = archDir / "gcnn_14B_lm_arch_as.txt";
  int nclass = 221452;
  int batchsize = 2;
  int inputlength = 100;
  std::vector<int> tail = {10000, 50000, 200000, nclass};

  auto model = buildSequentialModule(archfile, 1, nclass);
  auto as = std::make_shared<fl::AdaptiveSoftMax>(4096, tail);
  auto criterion = std::make_shared<fl::AdaptiveSoftMaxLoss>(as);
  model->eval();
  criterion->eval();
  auto input = fl::arange({inputlength, batchsize});
  auto output = model->forward(noGrad(input));
  output = as->forward(output);

  ASSERT_EQ(output.shape(), Shape({nclass, inputlength, batchsize}));

  // batchsize = 1
  batchsize = 1;
  input = fl::arange({inputlength, batchsize});
  output = model->forward(noGrad(input));
  output = as->forward(output);
  ASSERT_EQ(output.shape(), Shape({nclass, inputlength, batchsize}));
}

TEST(ConvLmModuleTest, GCNN14BCrossEntropy) {
  const fs::path archfile = archDir / "gcnn_14B_lm_arch_ce.txt";
  int nclass = 30;
  int batchsize = 2;
  int inputlength = 100;

  auto model = buildSequentialModule(archfile, 1, nclass);
  model->eval();
  auto input = fl::arange({inputlength, batchsize});
  auto output = model->forward(noGrad(input));
  ASSERT_EQ(output.shape(), Shape({nclass, inputlength, batchsize}));

  // batchsize = 1
  batchsize = 1;
  input = fl::arange({inputlength, batchsize});
  output = model->forward(noGrad(input));
  ASSERT_EQ(output.shape(), Shape({nclass, inputlength, batchsize}));
}

TEST(ConvLmModuleTest, SerializationGCNN14BAdaptiveSoftmax) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const fs::path path = fs::temp_directory_path() / "test.mdl";
  const fs::path archfile = archDir / "gcnn_14B_lm_arch_as.txt";

  int nclass = 221452;
  int batchsize = 2;
  int inputlength = 10;
  std::vector<int> tail = {10000, 50000, 200000, nclass};

  std::shared_ptr<fl::Module> model =
      buildSequentialModule(archfile, 1, nclass);
  auto as = std::make_shared<fl::AdaptiveSoftMax>(4096, tail);
  std::shared_ptr<BinaryModule> criterion =
      std::make_shared<fl::AdaptiveSoftMaxLoss>(as);
  model->eval();
  criterion->eval();
  auto input = noGrad(fl::arange({inputlength, batchsize}));
  auto output = model->forward({input})[0];
  auto output_criterion =
      std::dynamic_pointer_cast<AdaptiveSoftMaxLoss>(criterion)
          ->getActivation()
          ->forward(output);

  save(path, model, criterion);

  std::shared_ptr<Module> loaded_model;
  std::shared_ptr<BinaryModule> loaded_criterion;
  load(path, loaded_model, loaded_criterion);

  auto outputl = loaded_model->forward({input})[0];
  auto outputl_criterion =
      std::dynamic_pointer_cast<AdaptiveSoftMaxLoss>(loaded_criterion)
          ->getActivation()
          ->forward(outputl);

  ASSERT_TRUE(allParamsClose(*loaded_model.get(), *model));
  ASSERT_TRUE(allParamsClose(*loaded_criterion.get(), *criterion));
  ASSERT_TRUE(allClose(outputl, output));
  ASSERT_TRUE(allClose(outputl_criterion, output_criterion));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();

// Resolve directory for arch
#ifdef DECODER_TEST_DATADIR
  archDir = fs::path(DECODER_TEST_DATADIR);
#endif

  return RUN_ALL_TESTS();
}
