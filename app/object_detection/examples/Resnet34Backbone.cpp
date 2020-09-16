/**
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <iomanip>

#include <gflags/gflags.h>

#include "flashlight/dataset/datasets.h"
#include "flashlight/meter/meters.h"
#include "flashlight/optim/optim.h"
#include "vision/dataset/ImagenetUtils.h"
#include "vision/dataset/Transforms.h"
#include "vision/dataset/Utils.h"
#include "vision/Resnet34Backbone.h"

DEFINE_string(data_dir, "/datasets01_101/imagenet_full_size/061417/", "Directory of imagenet data");
DEFINE_double(lr, 0.1f, "Learning rate");
DEFINE_double(momentum, 0.9f, "Momentum");

DEFINE_double(wd, 1e-4f, "Weight decay");
DEFINE_uint64(epochs, 300, "Epochs");
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
    "/tmp/",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");
DEFINE_uint64(batch_size, 256, "Total batch size across all gpus");
DEFINE_string(checkpointpath, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(checkpoint, -1, "Load from checkpoint");

using namespace fl;
using namespace cv::dataset;

std::tuple<double, double, double> eval_loop(
    std::shared_ptr<Module> model,
    Dataset& dataset) {
  AverageValueMeter loss_meter;
  TopKMeter top5_meter(5, true);
  TopKMeter top1_meter(1, true);

  // Place the model in eval mode.
  model->eval();
  int idx = 0;
  for (auto& example : dataset) {
    auto inputs = noGrad(example[cv::dataset::INPUT_IDX]);
    auto output = model->forward({ inputs })[0];

    auto target = noGrad(example[cv::dataset::TARGET_IDX]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    loss_meter.add(loss.array().scalar<float>());
    top5_meter.add(output.array(), target.array());
    top1_meter.add(output.array(), target.array());
    idx++;
  }
  model->train();

  double top1_error = top1_meter.value();
  double top5_error = top5_meter.value();
  double loss = loss_meter.value()[0];
  return std::make_tuple(loss, top5_error, top1_error);
};


int main(int argc, char** argv) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string label_path = FLAGS_data_dir + "labels.txt";
  const std::string train_list = FLAGS_data_dir + "train";
  const std::string val_list = FLAGS_data_dir + "val";

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  af::info();
  fl::distributedInit(
	fl::DistributedInit::FILE_SYSTEM,
	FLAGS_world_rank,
	FLAGS_world_size,
	{{fl::DistributedConstants::kMaxDevicePerNode,
	  std::to_string(8)},
	 {fl::DistributedConstants::kFilePath, FLAGS_rndv_filepath}});

  std::cout << "WorldRank " << FLAGS_world_rank << " world_size " << FLAGS_world_size << std::endl;
  af::setDevice(FLAGS_world_rank);
  af::setSeed(FLAGS_world_size);

  auto reducer = std::make_shared<fl::CoalescingReducer>(
      1.0 / FLAGS_world_size,
      true,
      true);

  //////////////////////////
  //  Create datasets
  /////////////////////////
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  std::vector<ImageTransform> train_transforms = {
      // randomly resize shortest side of image between 256 to 480 for scale 
      // invariance
      randomResizeTransform(256, 480),
      randomCropTransform(224, 224),
      normalizeImage(mean, std),
      // Randomly flip image with probability of 0.5
      horizontalFlipTransform(0.5)
  };
  std::vector<ImageTransform> val_transforms = {
      // Resize shortest side to 256, then take a center crop
      resizeTransform(256),
      centerCropTransform(224),
      normalizeImage(mean, std)
  };

  const int64_t batch_size_per_gpu = FLAGS_batch_size / FLAGS_world_size;
  const int64_t prefetch_threads = 10;
  const int64_t prefetch_size = FLAGS_batch_size;
  auto train_ds = DistributedDataset(
      cv::dataset::imagenet(train_list, train_transforms),
      FLAGS_world_rank,
      FLAGS_world_size,
      batch_size_per_gpu,
      prefetch_threads,
      prefetch_size);

  auto val_ds = DistributedDataset(
      cv::dataset::imagenet(val_list, val_transforms),
      FLAGS_world_rank,
      FLAGS_world_size,
      batch_size_per_gpu,
      prefetch_threads,
      prefetch_size);

  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  //auto model = std::make_shared<Sequential>(resnet34());
  std::shared_ptr<Module>  model = std::make_shared<cv::Resnet34Backbone>();
  // synchronize parameters of tje model so that the parameters in each process
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
  auto saveModel = [&model](int epoch) {
    if(FLAGS_world_rank == 0) {
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
  if (FLAGS_checkpoint >= 0) {
    loadModel(FLAGS_checkpoint);
  }

  // The main training loop
  TimeMeter time_meter;
  TopKMeter top5_meter(5, true);
  TopKMeter top1_meter(1, true);
  AverageValueMeter train_loss_meter;
  for (int e = (FLAGS_checkpoint + 1); e < FLAGS_epochs; e++) {
    train_ds.resample();
    lrScheduler(e);

    // Get an iterator over the data
    time_meter.resume();
    int idx = 0;
    for (auto& example : train_ds) {
      opt.zeroGrad();
      // Make a Variable from the input array.
      auto input = noGrad(example[cv::dataset::INPUT_IDX]);

      // Get the activations from the model.
      auto output = model->forward({ input })[0];

      // Make a Variable from the target array.
      auto target = noGrad(example[cv::dataset::TARGET_IDX]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);

      train_loss_meter.add(loss.array());
      top5_meter.add(output.array(), target.array());
      top1_meter.add(output.array(), target.array());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();

      reducer->finalize();
      opt.step();

      // Compute and record the prediction error.
      double train_loss = train_loss_meter.value()[0];
      if (++idx % 50 == 0) {
        double time = time_meter.value();
        double sample_per_second = (idx * FLAGS_batch_size) / time;
        std::cout << "Epoch " << e << std::setprecision(5) << " Batch: " << idx
                  << " Samples per second " << sample_per_second
                  << ": Avg Train Loss: " << train_loss
                  << ": Train Top5 Error( %): " << top5_meter.value()
                  << ": Train Top1 Error( %): " << top1_meter.value()
                  << std::endl;
        top5_meter.reset();
        top1_meter.reset();
        train_loss_meter.reset();
      }
    }
    time_meter.reset();
    time_meter.stop();

    double val_loss, val_top1_err, val_top5_err;
    std::tie(val_loss, val_top5_err, val_top1_err) = eval_loop(model, val_ds);

    std::cout << "Epoch " << e << std::setprecision(5)
              << " Validation Loss: " << val_loss
              << " Validation Top5 Error (%): " << val_top5_err
              << " Validation Top1 Error (%): " << val_top1_err << std::endl;
    saveModel(e);
  }
}
