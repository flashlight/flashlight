/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <exception>
#include <iomanip>

#include <gflags/gflags.h>

#include "flashlight/app/image_classification/dataset/Imagenet.h"
#include "flashlight/dataset/datasets.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"
#include "flashlight/ext/image/fl/models/Resnet.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/meter/meters.h"
#include "flashlight/optim/optim.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_double(lr, 0.1f, "Learning rate");
DEFINE_double(momentum, 0.9f, "Momentum");

DEFINE_double(wd, 1e-4f, "Weight decay");
DEFINE_uint64(epochs, 50, "Number of epochs to train");
DEFINE_bool(enable_distributed, true, "Enable distributed training");
DEFINE_int64(
    max_devices_per_node,
    8,
    "the maximum number of devices per training node");
DEFINE_int64(
    world_rank,
    0,
    "rank of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    world_size,
    1,
    "total number of the process (Used if rndv_filepath is not empty)");
DEFINE_string(
    rndv_filepath,
    "",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");
DEFINE_uint64(batch_size, 256, "Total batch size across all gpus");
DEFINE_string(checkpointpath, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(checkpointepoch, -1, "Checkpoint epoch to load from");


using namespace fl;
using namespace fl::ext::image;
using namespace fl::app::image_classification;

// Returns the average loss, top 5 error, and top 1 error
std::tuple<double, double, double> evalLoop(
    std::shared_ptr<Sequential> model,
    Dataset& dataset) {
  AverageValueMeter lossMeter;
  TopKMeter top5Meter(5, true);
  TopKMeter top1Meter(1, true);

  // Place the model in eval mode.
  model->eval();
  int idx = 0;
  for (auto& example : dataset) {
    auto inputs = noGrad(example[kImagenetInputIdx]);
    auto output = model->forward(inputs);

    auto target = noGrad(example[kImagenetTargetIdx]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    lossMeter.add(loss.array().scalar<float>());
    top5Meter.add(output.array(), target.array());
    top1Meter.add(output.array(), target.array());
    idx++;
  }
  model->train();

  double top1Error = top1Meter.value();
  double top5Error = top5Meter.value();
  double loss = lossMeter.value()[0];
  return std::make_tuple(loss, top5Error, top1Error);
};


int main(int argc, char** argv) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string labelPath = lib::pathsConcat(FLAGS_data_dir, "labels.txt");
  const std::string trainList = lib::pathsConcat(FLAGS_data_dir,"train");
  const std::string valList = lib::pathsConcat(FLAGS_data_dir, "val");

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  af::info();
  if (FLAGS_enable_distributed) {
    fl::ext::initDistributed(
      FLAGS_world_rank,
      FLAGS_world_size,
      FLAGS_max_devices_per_node,
      FLAGS_rndv_filepath);
  }
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();
  const bool isMaster = (worldRank == 0);

  af::setDevice(worldRank);
  af::setSeed(worldSize);

  auto reducer = std::make_shared<fl::CoalescingReducer>(
      1.0 / worldSize,
      true,
      true);

  //////////////////////////
  //  Create datasets
  /////////////////////////
  // These are the mean and std for each channel of Imagenet
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const int randomResizeMax = 480;
  const int randomResizeMin = 256;
  const int randomCropSize = 224;
  const float horizontalFlipProb = 0.5f;
  std::vector<ImageTransform> trainTransforms = {
      // randomly resize shortest side of image between 256 to 480 for scale
      // invariance
      randomResizeTransform(randomResizeMin, randomResizeMax),
      randomCropTransform(randomCropSize, randomCropSize),
      normalizeImage(mean, std),
      // Randomly flip image with probability of 0.5
      horizontalFlipTransform(horizontalFlipProb)
  };
  std::vector<ImageTransform> valTransforms = {
      // Resize shortest side to 256, then take a center crop
      resizeTransform(randomResizeMin),
      centerCropTransform(randomCropSize),
      normalizeImage(mean, std)
  };

  const int64_t batchSizePerGpu = FLAGS_batch_size / worldSize;
  const int64_t prefetchThreads = 10;
  const int64_t prefetchSize = FLAGS_batch_size;
  auto trainDataset = DistributedDataset(
      imagenet(trainList, trainTransforms),
      worldRank,
      worldSize,
      batchSizePerGpu,
      prefetchThreads,
      prefetchSize);

  auto valDataset = DistributedDataset(
      imagenet(valList, valTransforms),
      worldRank,
      worldSize,
      batchSizePerGpu,
      prefetchThreads,
      prefetchSize);

  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  auto model = std::make_shared<Sequential>(resnet34());
  // synchronize parameters of the model so that the parameters in each process
  // is the same
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);

  SGDOptimizer opt(model->params(), FLAGS_lr, FLAGS_momentum, FLAGS_wd);

  auto lrScheduler = [&opt](int epoch) {
    // Adjust learning rate every 30 epoch
    if (epoch == 60 || epoch == 90 || epoch == 120) {
      const float newLr = opt.getLr() * 0.1;
      std::cout << "Setting learning rate to: " << newLr << std::endl;
      opt.setLr(newLr);
    }
  };

  // Small utility functions to load and save models
  auto saveModel = [&model, &isMaster](int epoch) {
    if(isMaster) {
      std::string modelPath = FLAGS_checkpointpath + std::to_string(epoch);
      std::cout <<  "Saving model to file: " << modelPath << std::endl;
      fl::save(modelPath, model);
    }
  };

  auto loadModel = [&model](int epoch) {
      std::string modelPath = FLAGS_checkpointpath + std::to_string(epoch);
      std::cout <<  "Loading model from file: " << modelPath << std::endl;
      fl::load(modelPath, model);
  };
  if (FLAGS_checkpointepoch >= 0) {
    loadModel(FLAGS_checkpointepoch);
  }

  // The main training loop
  TimeMeter timeMeter;
  TopKMeter top5Meter(5, true);
  TopKMeter top1Meter(1, true);
  AverageValueMeter trainLossMeter;
  for (int e = (FLAGS_checkpointepoch + 1); e < FLAGS_epochs; e++) {
    trainDataset.resample();
    lrScheduler(e);

    // Get an iterator over the data
    timeMeter.resume();
    int idx = 0;
    for (auto& example : trainDataset) {
      opt.zeroGrad();
      // Make a Variable from the input array.
      auto inputs = noGrad(example[kImagenetInputIdx]);

      // Get the activations from the model.
      auto output = model->forward(inputs);

      // Make a Variable from the target array.
      auto target = noGrad(example[kImagenetTargetIdx]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);

      trainLossMeter.add(loss.array());
      top5Meter.add(output.array(), target.array());
      top1Meter.add(output.array(), target.array());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();

      if(FLAGS_enable_distributed) {
        reducer->finalize();
      }
      opt.step();

      // Compute and record the prediction error.
      double trainLoss = trainLossMeter.value()[0];
      if (++idx % 50 == 0) {
        double time = timeMeter.value();
        double samplePerSecond = (idx * FLAGS_batch_size) / time;
        std::cout << "Epoch " << e << std::setprecision(5) << " Batch: " << idx
                  << " Samples per second " << samplePerSecond
                  << ": Avg Train Loss: " << trainLoss
                  << ": Train Top5 Error( %): " << top5Meter.value()
                  << ": Train Top1 Error( %): " << top1Meter.value()
                  << std::endl;
        top5Meter.reset();
        top1Meter.reset();
        trainLossMeter.reset();
      }
    }
    timeMeter.reset();
    timeMeter.stop();

    double valLoss, valTop1Error, valTop5Err;
    std::tie(valLoss, valTop5Err, valTop1Error) = evalLoop(model, valDataset);

    std::cout << "Epoch " << e << std::setprecision(5)
              << " Validation Loss: " << valLoss
              << " Validation Top5 Error (%): " << valTop5Err
              << " Validation Top1 Error (%): " << valTop1Error << std::endl;
    saveModel(e);
  }
  std::cout << "Training complete" << std::endl;
}
