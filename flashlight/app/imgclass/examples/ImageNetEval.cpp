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

#include "flashlight/pkg/vision/dataset/Imagenet.h"
#include "flashlight/app/imgclass/examples/Defines.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/pkg/vision/dataset/Transforms.h"
#include "flashlight/pkg/vision/dataset/DistributedDataset.h"
#include "flashlight/pkg/vision/models/Resnet.h"
#include "flashlight/pkg/vision/models/ViT.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/lib/common/System.h"

#include "flashlight/fl/common/threadpool/ThreadPool.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_uint64(data_batch_size, 256, "Batch size per gpus");
DEFINE_string(exp_checkpoint_path, "/tmp/model", "Checkpointing prefix path");

DEFINE_bool(distributed_enable, true, "Enable distributed evaluation");
DEFINE_int64(
    distributed_max_devices_per_node,
    8,
    "the maximum number of devices per training node");
DEFINE_int64(
    distributed_world_rank,
    0,
    "rank of the process (Used if distributed_rndv_filepath is not empty)");
DEFINE_int64(
    distributed_world_size,
    1,
    "total number of the process (Used if distributed_rndv_filepath is not empty)");
DEFINE_string(
    distributed_rndv_filepath,
    "",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");

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

  if (FLAGS_distributed_enable) {
    fl::ext::initDistributed(
        FLAGS_distributed_world_rank,
        FLAGS_distributed_world_size,
        FLAGS_distributed_max_devices_per_node,
        FLAGS_distributed_rndv_filepath);
  }
  af::info();
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();

  std::shared_ptr<fl::Module> model;
  fl::load(FLAGS_exp_checkpoint_path, model);

  const std::string labelPath = lib::pathsConcat(FLAGS_data_dir, "labels.txt");
  const std::string testList = lib::pathsConcat(FLAGS_data_dir, "val");

  //  Create datasets
  FL_LOG_MASTER(INFO) << "Creating dataset";
  // TODO: only support training with image shape 224 x 224 in this example
  const int imageSize = 224;
  // Conventional image resize parameter used for evaluation
  const int randomResizeMin = imageSize / .875;
  ImageTransform testTransforms = compose(
      {fl::ext::image::resizeTransform(randomResizeMin),
       fl::ext::image::centerCropTransform(imageSize),
       fl::ext::image::normalizeImage(
           fl::app::image::kImageNetMean, fl::app::image::kImageNetStd)});

  auto labelMap = getImagenetLabels(labelPath);
  auto testDataset = fl::ext::image::DistributedDataset(
      imagenetDataset(testList, labelMap, {testTransforms}),
      worldRank,
      worldSize,
      FLAGS_data_batch_size,
      1, // train_n_repeatedaug
      10, // prefetch threads
      FLAGS_data_batch_size,
      fl::BatchDatasetPolicy::INCLUDE_LAST);
  FL_LOG_MASTER(INFO) << "[testDataset size] " << testDataset.size();

  // The main evaluation loop
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);

  model->eval();
  for (auto& example : testDataset) {
    auto inputs = noGrad(example[kImagenetInputIdx]);
    auto output = model->forward({inputs}).front();
    auto target = noGrad(example[kImagenetTargetIdx]);

    top5Acc.add(output.array(), target.array());
    top1Acc.add(output.array(), target.array());
  }
  fl::ext::syncMeter(top5Acc);
  fl::ext::syncMeter(top1Acc);

  FL_LOG_MASTER(INFO) << "Top 5 acc: " << top5Acc.value();
  FL_LOG_MASTER(INFO) << "Top 1 acc: " << top1Acc.value();
}
