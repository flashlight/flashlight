/**
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 * * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <iomanip>

#include <cudnn.h>

#include "flashlight/contrib/modules/Resnet.h"
#include "flashlight/dataset/BatchDataset.h"
#include "flashlight/dataset/Dataset.h"
#include "flashlight/dataset/ImagenetUtils.h"
#include "flashlight/dataset/PrefetchDataset.h"
#include "flashlight/dataset/ShuffleDataset.h"
#include "flashlight/meter/AverageValueMeter.h"
#include "flashlight/meter/TimeMeter.h"
#include "flashlight/meter/TopKMeter.h"
#include "flashlight/nn/nn.h"
#include "flashlight/optim/optim.h"

#include <gflags/gflags.h>

DEFINE_string(data_dir, "/datasets01_101/imagenet_full_size/061417/", "Directory of imagenet data");
DEFINE_double(lr, 0.1f, "Learning rate");
DEFINE_double(momentum, 0.9f, "Momentum");

DEFINE_double(wd, 1e-4f, "Weight decay");
DEFINE_uint64(epochs, 50, "Epochs");
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
DEFINE_int64(checkpoint, -1, "Load from checkpoint");

using namespace fl;

class DistributedDataset : public Dataset {
 public:
  DistributedDataset(
      std::shared_ptr<Dataset> base,
      int64_t world_rank,
      int64_t world_size,
      int64_t batch_size,
      int64_t num_threads,
      int64_t prefetch_size,
      bool shuffle) {
    ds_ = base;
    if (shuffle) {
      shuffle_ = std::make_shared<ShuffleDataset>(ds_);
      ds_ = shuffle_;
    }

    auto permfn = [world_size, world_rank, &base](int64_t idx) {
      return (idx * world_size) + world_rank;
    };
    ds_ = std::make_shared<ResampleDataset>(ds_, permfn, ds_->size() / world_size);

    if (num_threads > 0) {
      ds_ = std::make_shared<PrefetchDataset>(ds_, num_threads, prefetch_size);
    }
    if (batch_size > 1) {
      ds_ = std::make_shared<BatchDataset>(ds_, batch_size);
    }
  }

  std::vector<af::array> get(const int64_t idx) const override {
    checkIndexBounds(idx);
    return ds_->get(idx);
  }

  void resample() {
    if (shuffle_) {
      shuffle_->resample();
    } else {
      std::cerr << " Dataset not build with shuffling!" << std::endl;
}
  }

  int64_t size() const override {
    return ds_->size();
  }

 private:
  std::shared_ptr<Dataset> ds_;
  std::shared_ptr<ShuffleDataset> shuffle_;
};

std::tuple<double, double, double> eval_loop(
    std::shared_ptr<Sequential> model,
    Dataset& dataset) {
  AverageValueMeter loss_meter;
  TopKMeter top5_meter(5, true);
  TopKMeter top1_meter(1, true);

  // Place the model in eval mode.
  model->eval();
  int idx = 0;
  for (auto& example : dataset) {
    auto inputs = noGrad(example[ImageDataset::INPUT_IDX]);
    auto output = model->forward(inputs);

    auto target = noGrad(example[ImageDataset::TARGET_IDX]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    //std::cout << " loss " << std::endl;
    loss_meter.add(loss.array().scalar<float>());
    top5_meter.add(output.array(), target.array());
    top1_meter.add(output.array(), target.array());
    idx++;
  }
  // Place the model back into train mode.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  double top1_error = top1_meter.value();
  double top5_error = top5_meter.value();
  double loss = loss_meter.value()[0];
  return std::make_tuple(loss, top5_error, top1_error);
};


int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Must specify imagenet data location" << std::endl;
    return -1;
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  af::setDevice(FLAGS_world_rank);

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

  af::setSeed(FLAGS_world_size);

  auto reducer = std::make_shared<fl::CoalescingReducer>(
      1.0 / FLAGS_world_size,
      true,
      true);


  //////////////////////////
  //  Create datasets
  /////////////////////////
  const int batch_size_per_gpu = FLAGS_batch_size / FLAGS_world_size;
  const int64_t prefetch_threads = 10;
  const int64_t prefetch_size = batch_size_per_gpu * 2;
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  auto labels = imagenetLabels(label_path);
  std::vector<Dataset::TransformFunction> train_transforms = {
      ImageDataset::randomResizeCropTransform(224, 0.08, 1.0, 3./.4, 4./3.),
      //// Randomly flip image with probability of 0.5
      ImageDataset::horizontalFlipTransform(0.5),
      ImageDataset::centerCrop(224),
      ImageDataset::normalizeImage(mean, std)
  };
  std::vector<Dataset::TransformFunction> val_transforms = {
      // Resize shortest side to 256, then take a center crop
      ImageDataset::resizeTransform(256),
      ImageDataset::centerCrop(224),
      ImageDataset::normalizeImage(mean, std)
  };
  auto test = std::make_shared<ImageDataset>(
          imagenetDataset(train_list, labels, train_transforms));
  auto train_ds = DistributedDataset(
      test,
      FLAGS_world_rank,
      FLAGS_world_size,
      batch_size_per_gpu,
      prefetch_threads,
      prefetch_size, true);

  auto test_val = std::make_shared<ImageDataset>(imagenetDataset(val_list, labels, val_transforms));
  auto val_ds = DistributedDataset(
      test_val,
      FLAGS_world_rank,
      FLAGS_world_size,
      batch_size_per_gpu,
      prefetch_threads,
      prefetch_size, false);


  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  auto model = std::make_shared<Sequential>(resnet34());

  SGDOptimizer opt(model->params(), FLAGS_lr, FLAGS_momentum, FLAGS_wd);

  auto lrScheduler = [&opt](int epoch) {
    // Adjust learning rate every 30 epoch
    if (epoch > 0 && epoch % 30 == 0) {
      const float newLr = opt.getLr() * 0.1;
      std::cout << "Setting learning rate to: " << newLr << std::endl;
      opt.setLr(newLr);
    }
  };

  // Small utility function to save models
  std::string checkpointPrefix = "/private/home/padentomasello/code/flashlight/build/model-";
  auto saveModel = [&model, &checkpointPrefix](int epoch) {
    if(FLAGS_world_rank == 0) {
      std::string modelPath = FLAGS_checkpointpath + std::to_string(epoch);
      std::cout <<  "Saving model to file: " << modelPath << std::endl;
      fl::save(modelPath, model);
    }
  };

  if (FLAGS_checkpoint >= 0) {
    std::string modelPath = checkpointPrefix + std::to_string(FLAGS_checkpoint);
    fl::load(modelPath, model);
  }

  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);

  // The main training loop
  TimeMeter time_meter;
  TopKMeter top5_meter(5, true);
  TopKMeter top1_meter(1, true);
  AverageValueMeter train_loss_meter;
  model->train();
  for (int e = (FLAGS_checkpoint + 1); e < FLAGS_epochs; e++) {
    train_ds.resample();
    lrScheduler(e);

    // Get an iterator over the data
    time_meter.resume();
    int idx = 0;
    for (auto& example : train_ds) {
      // Make a Variable from the input array.
      auto inputs = noGrad(example[ImageDataset::INPUT_IDX]);

      // Get the activations from the model.
      auto output = model->forward(inputs);

      // Make a Variable from the target array.
      auto target = noGrad(example[ImageDataset::TARGET_IDX]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);
      //std::cout << loss << std::endl;

      train_loss_meter.add(loss.array());
      top5_meter.add(output.array(), target.array());
      top1_meter.add(output.array(), target.array());

      // Backprop, update the weights and then zero the gradients.
      reducer->finalize();
      loss.backward();
      opt.step();
      opt.zeroGrad();

      // Compute and record the prediction error.
      if (++idx % 10 == 0) {
        double train_loss = train_loss_meter.value()[0];
        double time = time_meter.value();
        double sample_per_second = ((idx * FLAGS_batch_size) / time);
        double top5 = top5_meter.value();
        double top1 = top1_meter.value();
        af::array train_loss_arr = af::array(1, &train_loss);
        af::array top1_arr = af::array(1, &top1);
        af::array top5_arr = af::array(1, &top5);
        af::array samples_per_second_arr = af::array(1, &sample_per_second);
        std::vector<af::array*> metric_arrays = {
          &train_loss_arr, &top1_arr, &top5_arr, &samples_per_second_arr
        };
        fl::allReduceMultiple(metric_arrays, false, false);
        if (FLAGS_world_rank == 0) {
          std::cout << "Epoch " << e << std::setprecision(6) << " Batch: " << idx
                    << " Samples per second " << samples_per_second_arr.scalar<double>()
                    << ": Avg Train Loss: " << train_loss_arr.scalar<double>() / FLAGS_world_size
                    << ": Train Top5 Error( %): " << top5_arr.scalar<double>() / FLAGS_world_size
                    << ": Train Top1 Error( %): " << top1_arr.scalar<double>() / FLAGS_world_size
                    << std::endl;
        }
      }
    }
    top5_meter.reset();
    top1_meter.reset();
    train_loss_meter.reset();
    time_meter.reset();
    time_meter.stop();

    // Evaluate on the dev set.
    double val_loss, val_top1_err, val_top5_err;
    std::tie(val_loss, val_top5_err, val_top1_err) = eval_loop(model, val_ds);

    std::cout << "Epoch " << e << std::setprecision(6)
              << " Validation Loss: " << val_loss
              << " Validation Top5 Error (%): " << val_top5_err
              << " Validation Top1 Error (%): " << val_top1_err << std::endl;
    //saveModel(e);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}
