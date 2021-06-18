/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include "flashlight/fl/flashlight.h"

#include "flashlight/fl/common/SequentialBuilder.h"
#include "flashlight/lib/common/System.h"

using namespace fl;
using namespace fl::lib;
using namespace fl::ext;

namespace {

std::string archDir = "";

} // namespace

TEST(ConvLmModuleTest, GCNN14BAdaptiveSoftmax) {
  const std::string archfile = pathsConcat(archDir, "gcnn_14B_lm_arch_as.txt");
  int nclass = 221452;
  int batchsize = 2;
  int inputlength = 100;
  std::vector<int> tail = {10000, 50000, 200000, nclass};

  auto model = buildSequentialModule(archfile, 1, nclass);
  auto as = std::make_shared<fl::AdaptiveSoftMax>(4096, tail);
  auto criterion = std::make_shared<fl::AdaptiveSoftMaxLoss>(as);
  model->eval();
  criterion->eval();
  auto input = af::range(af::dim4(inputlength, batchsize), f32);
  auto output = model->forward(noGrad(input));
  output = as->forward(output);

  ASSERT_EQ(output.dims(), af::dim4(nclass, inputlength, batchsize));

  // batchsize = 1
  batchsize = 1;
  input = af::range(af::dim4(inputlength), f32);
  output = model->forward(noGrad(input));
  output = as->forward(output);
  ASSERT_EQ(output.dims(), af::dim4(nclass, inputlength, batchsize));
}

TEST(ConvLmModuleTest, GCNN14BCrossEntropy) {
  const std::string archfile = pathsConcat(archDir, "gcnn_14B_lm_arch_ce.txt");
  int nclass = 30;
  int batchsize = 2;
  int inputlength = 100;

  auto model = buildSequentialModule(archfile, 1, nclass);
  model->eval();
  auto input = af::range(af::dim4(inputlength, batchsize), f32);
  auto output = model->forward(noGrad(input));
  ASSERT_EQ(output.dims(), af::dim4(nclass, inputlength, batchsize));

  // batchsize = 1
  batchsize = 1;
  input = af::range(af::dim4(inputlength), f32);
  output = model->forward(noGrad(input));
  ASSERT_EQ(output.dims(), af::dim4(nclass, inputlength, batchsize));
}

TEST(ConvLmModuleTest, SerializationGCNN14BAdaptiveSoftmax) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const std::string path = getTmpPath("test.mdl");
  const std::string archfile = pathsConcat(archDir, "gcnn_14B_lm_arch_as.txt");

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
  auto input = noGrad(af::range(af::dim4(inputlength, batchsize), f32));
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
          ->forward(output);

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
  archDir = DECODER_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
