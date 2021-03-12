/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <exception>
#include <iomanip>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/app/imgclass/dataset/Imagenet.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"
#include "flashlight/ext/image/fl/models/Resnet.h"
#include "flashlight/ext/image/fl/models/ViT.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/lib/common/System.h"

#include "flashlight/fl/common/threadpool/ThreadPool.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_uint64(data_batch_size, 256, "Batch size per gpus");
DEFINE_string(exp_checkpoint_path, "/tmp/model", "Checkpointing prefix path");

using namespace fl;
using fl::ext::image::compose;
using fl::ext::image::ImageTransform;
using namespace fl::app::imgclass;

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

int main(int argc, char** argv) {
  fl::init();
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  fl::ext::initDistributed(0, 1, 8, "");
  af::info();
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();
  const bool isMaster = (worldRank == 0);

  // std::shared_ptr<fl::ext::image::ViT> model;
  // fl::load(FLAGS_exp_checkpoint_path, model);

  auto model = std::make_shared<fl::ext::image::ViT>(FLAGS_exp_checkpoint_path);
#if 0
  std::ifstream fin(
      "/private/home/qiantong/tmp/vitb_pt/sample.bin", std::ios::binary);
  if (!fin) {
    throw std::runtime_error(" Error, Couldn't find the file");
  }

  fin.seekg(0, std::ios::end);
  const size_t num_elements = fin.tellg() / sizeof(float);
  fin.seekg(0, std::ios::beg);

  std::vector<float> res(num_elements);
  fin.read(reinterpret_cast<char*>(res.data()), num_elements * sizeof(float));

  if (res.size() != 9633792) {
    throw std::runtime_error("wrong size");
  }
  auto var = fl::Variable(af::array(af::dim4(224, 224, 3, 64), res.data()));
  // af_print(var.array()(af::seq(0, 19), 0, 0, 0));
  // af_print(var.array()(0, 0, af::span, 0));

  model->eval();
  auto out = model->forward({var}).front();
  // return 0;
  std::cout << out.dims() << std::endl;
  af_print(out.array()(af::seq(0, 19), 0));

  af::array randMatrixSorted, randMatrixSortedIndices;
  // create random permutation
  af::sort(
      randMatrixSorted, randMatrixSortedIndices, out.array().as(f32), 0, false);
  std::cout << randMatrixSortedIndices.dims() << std::endl;
  std::cout << randMatrixSorted.dims() << std::endl;

  af_print(randMatrixSortedIndices(0, af::seq(0, 19)));
  af_print(randMatrixSorted(0, af::seq(0, 19)));

  return 0;
#endif
  const std::string labelPath = lib::pathsConcat(FLAGS_data_dir, "labels.txt");
  const std::string testList = lib::pathsConcat(FLAGS_data_dir, "val");

  //  Create datasets
  FL_LOG_MASTER(INFO) << "Creating dataset";
  // These are the mean and std for each channel of Imagenet
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const int randomResizeMax = 480;
  const int randomResizeMin = 256;
  const int randomCropSize = 224;
  ImageTransform testTransforms =
      compose({// Resize shortest side to 256, then take a center crop
               fl::ext::image::resizeTransform(randomResizeMin),
               fl::ext::image::centerCropTransform(randomCropSize),
               fl::ext::image::normalizeImage(mean, std)});

  auto labelMap = getImagenetLabels(labelPath);
  auto testDataset = fl::ext::image::DistributedDataset(
      imagenetDataset(testList, labelMap, {testTransforms}),
      worldRank,
      worldSize,
      FLAGS_data_batch_size,
      1,
      10,
      FLAGS_data_batch_size,
      fl::BatchDatasetPolicy::INCLUDE_LAST);
  FL_LOG_MASTER(INFO) << "[testDataset size] " << testDataset.size();

  // The main training loop
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);

  // Place the model in eval mode.
  model->eval();
  for (auto& example : testDataset) {
    auto inputs = noGrad(example[kImagenetInputIdx]);
    auto output = model->forward({inputs}).front();
    auto target = noGrad(example[kImagenetTargetIdx]);

    // Compute and record the loss.
    top5Acc.add(output.array(), target.array());
    top1Acc.add(output.array(), target.array());
  }
  fl::ext::syncMeter(top5Acc);
  fl::ext::syncMeter(top1Acc);

  FL_LOG_MASTER(INFO) << "Top 5 acc: " << top5Acc.value();
  FL_LOG_MASTER(INFO) << "Top 1 acc: " << top1Acc.value();
}
