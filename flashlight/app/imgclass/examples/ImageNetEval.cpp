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
#include "flashlight/app/imgclass/examples/Defines.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"
#include "flashlight/ext/image/fl/models/Resnet.h"
#include "flashlight/ext/image/fl/models/ViT.h"
#include "flashlight/ext/image/fl/nn/VisionTransformer.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/lib/common/System.h"

#include "flashlight/fl/common/threadpool/ThreadPool.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_string(data_path, "val", "Directory of data part");
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
DEFINE_int64(image_size, 224, "image size to test on");
DEFINE_double(region, 1, "normlization to grid in the sinpos");
DEFINE_int64(shiftX, 0, "shift image before eval");
DEFINE_int64(shiftY, 0, "shift image before eval");
DEFINE_bool(is_mnist, false, "mnist data");
DEFINE_bool(use_mix, false, "eval using several scales");
DEFINE_bool(use_own_size, false, "eval using own size");
DEFINE_bool(nocrop, false, "eval using own size + no crop");
DEFINE_bool(speed, false, "eval speed");
DEFINE_bool(visualize, false, "print for visualization");
DEFINE_bool(
    fl_amp_use_mixed_precision,
    false,
    "[train] Use mixed precision for training - scale loss and gradients up and down "
    "by a scale factor that changes over time. If no fl optim mode is "
    "specified with --fl_optim_mode when passing this flag, automatically "
    "sets the optim mode to O1.");

using namespace fl;
using fl::ext::image::compose;
using fl::ext::image::ImageTransform;
using namespace fl::app::imgclass;

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

ImageTransform resizeCenterCropOwnTransform(const float scale) {
  return [scale](const af::array& in) {
    if (scale == 1) {
      return in;
    }
    const int sizew = in.dims(0);
    const int sizeh = in.dims(1);
    const int sizewNew = in.dims(0) / scale;
    const int sizehNew = in.dims(1) / scale;
    auto result = af::resize(in, sizewNew, sizehNew, AF_INTERP_BILINEAR);
    const int cropLeft =
        std::round((static_cast<float>(sizewNew) - sizew) / 2.);
    const int cropTop = std::round((static_cast<float>(sizehNew) - sizeh) / 2.);
    result = fl::ext::image::crop(result, cropLeft, cropTop, sizew, sizeh);
    return result;
  };
}

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
  fl::DynamicBenchmark::setBenchmarkMode(true);
  af::info();
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();

  std::shared_ptr<fl::ext::image::ViT> model;
  // std::shared_ptr<fl::Sequential> model;
  fl::load(FLAGS_exp_checkpoint_path, model);
  const int imageSize = FLAGS_image_size;
  model->resetPosEmb(imageSize / 16);
  if (FLAGS_visualize) {
    model->train();
    af::array inputs = af::randn(af::dim4(imageSize, imageSize, 3, 1));
    auto output = model
                      ->forward(
                          {fl::noGrad(inputs)},
                          FLAGS_fl_amp_use_mixed_precision,
                          true,
                          false,
                          FLAGS_region)
                      .front();
    output = model
                 ->forward(
                     {fl::noGrad(inputs)},
                     FLAGS_fl_amp_use_mixed_precision,
                     true,
                     true,
                     FLAGS_region)
                 .front();
    return 0;
  }

  model->eval();
  LOG(INFO) << "Run evaluation";
  if (FLAGS_speed) {
    TimeMeter timeMeter;
    af::array inputs =
        af::randn(af::dim4(imageSize, imageSize, 3, FLAGS_data_batch_size));
    for (int tind = 0; tind < 30; tind++) {
      auto output = model
                        ->forward(
                            {fl::noGrad(inputs)},
                            FLAGS_fl_amp_use_mixed_precision,
                            true,
                            false,
                            FLAGS_region)
                        .front();
    }
    timeMeter.resume();
    for (int tind = 0; tind < 30; tind++) {
      auto output = model
                        ->forward(
                            {fl::noGrad(inputs)},
                            FLAGS_fl_amp_use_mixed_precision,
                            true,
                            false,
                            FLAGS_region)
                        .front();
    }
    timeMeter.stop();
    FL_LOG_MASTER(INFO) << "Througput img/s: "
                        << FLAGS_data_batch_size / timeMeter.value() * 30;
    return 0;
  }

  const std::string labelPath = lib::pathsConcat(FLAGS_data_dir, "labels.txt");
  const std::string testList =
      lib::pathsConcat(FLAGS_data_dir, FLAGS_data_path);

  if (FLAGS_fl_amp_use_mixed_precision) {
    FL_LOG_MASTER(INFO)
        << "Mixed precision training enabled. Will perform loss scaling.";
    // TODO: force using `DEFAULT` level for now, until
    //  1. comprehensive benchmarking
    //  2. AMP config updated accordingly
    fl::OptimMode::get().setOptimLevel(fl::OptimLevel::DEFAULT);
  }
  //  Create datasets
  FL_LOG_MASTER(INFO) << "Creating dataset";
  // Conventional image resize parameter used for evaluation
  const int randomResizeMin = imageSize / .875;
  std::vector<float> dataMean = fl::app::image::kImageNetMean,
                     dataStd = fl::app::image::kImageNetStd;
  if (FLAGS_is_mnist) {
    dataMean = {0};
    dataStd = {255};
  }

  std::vector<ImageTransform> testTransforms = {compose({
      fl::ext::image::resizeTransform(randomResizeMin),
      fl::ext::image::centerCropTransform(imageSize),
      fl::ext::image::normalizeImage(dataMean, dataStd),
      fl::ext::image::translateXtr(FLAGS_shiftX),
      fl::ext::image::translateYtr(FLAGS_shiftY)
      // fl::ext::image::normalizeImage(dataMean, dataStd)
  })};
  if (FLAGS_use_own_size) {
    float scale = FLAGS_nocrop ? 1 : 0.875;
    testTransforms = {compose({
        resizeCenterCropOwnTransform(scale),
        fl::ext::image::normalizeImage(dataMean, dataStd),
        fl::ext::image::translateXtr(FLAGS_shiftX),
        fl::ext::image::translateYtr(FLAGS_shiftY)
        // fl::ext::image::normalizeImage(dataMean, dataStd)
    })};
  }
  if (FLAGS_is_mnist) {
    testTransforms = {compose(
        {fl::ext::image::resizeTransform(imageSize),
         fl::ext::image::normalizeImage(dataMean, dataStd),
         fl::ext::image::translateXtr(FLAGS_shiftX),
         fl::ext::image::translateYtr(FLAGS_shiftY)})};
  }

  auto labelMap = getImagenetLabels(labelPath);
  std::vector<fl::ext::image::DistributedDataset> testDataset = {
      fl::ext::image::DistributedDataset(
          imagenetDataset(testList, labelMap, {testTransforms.back()}),
          worldRank,
          worldSize,
          FLAGS_data_batch_size,
          1, // train_n_repeatedaug
          1, // prefetch threads
          FLAGS_data_batch_size,
          fl::BatchDatasetPolicy::INCLUDE_LAST)};
  FL_LOG_MASTER(INFO) << "[testDataset size] " << testDataset.back().size();

  if (FLAGS_use_mix) {
    std::vector<int> sizes = {imageSize - 32, imageSize + 32};
    for (auto s : sizes) {
      testTransforms.push_back(compose(
          {fl::ext::image::resizeTransform(s / .875),
           fl::ext::image::centerCropTransform(s),
           fl::ext::image::normalizeImage(dataMean, dataStd),
           fl::ext::image::translateXtr(FLAGS_shiftX),
           fl::ext::image::translateYtr(FLAGS_shiftY)}));
      testDataset.push_back(fl::ext::image::DistributedDataset(
          imagenetDataset(testList, labelMap, {testTransforms.back()}),
          worldRank,
          worldSize,
          FLAGS_data_batch_size,
          1, // train_n_repeatedaug
          10, // prefetch threads
          FLAGS_data_batch_size,
          fl::BatchDatasetPolicy::INCLUDE_LAST));
    }
  }
  // The main evaluation loop
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);
  AverageValueMeter lossMeter;

  for (auto index = 0; index < testDataset[0].size(); index++) {
    fl::Variable target, outputTotal;
    for (int dataIndex = 0; dataIndex < testDataset.size(); dataIndex++) {
      auto example = testDataset[dataIndex].get(index);
      auto inputs = noGrad(example[kImagenetInputIdx]);
      auto output = model
                        ->forward(
                            {inputs},
                            FLAGS_fl_amp_use_mixed_precision,
                            true,
                            false,
                            FLAGS_region)
                        .front();
      if (dataIndex == 0) {
        outputTotal = logSoftmax(output, 0).as(output.type());
      } else {
        outputTotal = outputTotal + logSoftmax(output, 0).as(output.type());
      }
      target = noGrad(example[kImagenetTargetIdx]);
      // if (FLAGS_speed) {
      //   TimeMeter timeMeter;
      //   timeMeter.resume();
      //   for (int tind = 0; tind < 30; tind++) {
      //     output = model
      //                  ->forward(
      //                      {inputs},
      //                      FLAGS_fl_amp_use_mixed_precision,
      //                      true,
      //                      false,
      //                      FLAGS_region)
      //                  .front();
      //     // outputTotal = logSoftmax(output, 0).as(output.type());
      //   }
      //   timeMeter.stop();
      //   FL_LOG_MASTER(INFO) << "Througput img/s: "
      //                       << FLAGS_data_batch_size / timeMeter.value() *
      //                       30;
      // }
    }
    outputTotal = logSoftmax(outputTotal, 0).as(outputTotal.type());
    auto loss = categoricalCrossEntropy(outputTotal, target);
    lossMeter.add(loss.array().scalar<float>());
    top5Acc.add(outputTotal.array(), target.array());
    top1Acc.add(outputTotal.array(), target.array());
    // if (FLAGS_speed) {
    //   break;
    // }
  }
  fl::ext::syncMeter(top5Acc);
  fl::ext::syncMeter(top1Acc);
  fl::ext::syncMeter(lossMeter);

  FL_LOG_MASTER(INFO) << "Loss : " << lossMeter.value()[0];
  FL_LOG_MASTER(INFO) << "Top 5 acc: " << top5Acc.value();
  FL_LOG_MASTER(INFO) << "Top 1 acc: " << top1Acc.value();
}
