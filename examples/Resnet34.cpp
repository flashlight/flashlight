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

DEFINE_double(lr, 0.1f, "Learning rate");
DEFINE_double(momentum, 0.9f, "Momentum");

DEFINE_double(wd, 1e-4f, "Weight decay");
DEFINE_uint64(epochs, 50, "Epochs");
DEFINE_uint64(world_rank, 0, "Epochs");
DEFINE_uint64(world_size, 1, "Epochs");
DEFINE_uint64(batch_size, 32, "Epochs");


#define DISTRIBUTED 0

#define TRAIN 1
#define CACHE 0

namespace {


}

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

    //shuffle_ = std::make_shared<ShuffleDataset>(base);
    //ds_ = shuffle_;
    //auto permfn = [world_size, world_rank, &base](int64_t idx) {
      //return (idx * 1187) % base->size();
    //};
    //ds_ = std::make_shared<ResampleDataset>(ds_, permfn, 1024);
    ds_ = std::make_shared<PrefetchDataset>(ds_, num_threads, prefetch_size);
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
  const std::string imagenet_base = argv[1];
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int world_rank = FLAGS_world_rank;
  int world_size = FLAGS_world_size;
  int miniBatchSize = FLAGS_batch_size;
  af::setDevice(world_rank);
  if (world_size > 1 && !DISTRIBUTED) {
    std::cout << "Not built for distributed!" << std::endl;
    return -1;
  }

  const std::string label_path = imagenet_base + "labels.txt";
  const std::string train_list = imagenet_base + "train";
  const std::string val_list = imagenet_base + "val";

  /////////////////////////
  // Hyperparaters
  ////////////////////////
  //const int batch_size = 256;
  //const int miniBatchSize = 128;
  const float learning_rate = FLAGS_lr;
  const float momentum = FLAGS_momentum;
  const float weight_decay = FLAGS_wd;
  const int epochs = FLAGS_epochs;

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  af::info();
#if DISTRIBUTED
  fl::distributedInit(
	fl::DistributedInit::FILE_SYSTEM,
	world_rank,
	world_size,
	{{fl::DistributedConstants::kMaxDevicePerNode,
	  std::to_string(8)},
	 {fl::DistributedConstants::kFilePath, "/checkpoint/padentomasello/tmp/" + std::to_string(world_size)}});

  std::cout << "WorldRank" << world_rank << "world_size " << world_size << std::endl;
  af::setSeed(world_size);

  auto reducer = std::make_shared<fl::CoalescingReducer>(
      1.0 / world_size,
      true,
      true);
#endif




  //////////////////////////
  //  Create datasets
  /////////////////////////
  const int batch_size = miniBatchSize * world_size;
  const int64_t prefetch_threads = 10;
  const int64_t prefetch_size = miniBatchSize * 2;
#if CACHE
  auto test = std::make_shared<NumpyDataset>(1024, "/private/home/padentomasello/tmp/pytorch_dump/save/train/");
  auto train_ds = DistributedDataset(
      test,
      world_rank,
      world_size,
      miniBatchSize,
      prefetch_threads,
      prefetch_size,
      false);

  auto test_val = std::make_shared<NumpyDataset>(128, "/private/home/padentomasello/tmp/pytorch_dump/save/val/");
  //test_val = std::make_shared<ImageDataset>(imagenetDataset(val_list, labels, val_transforms);
  auto val_ds = DistributedDataset(
      test_val,
      world_rank,
      world_size,
      miniBatchSize,
      prefetch_threads,
      prefetch_size,
      false);
#else
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  //const std::vector<float> mean = {0.406, 0.456, 0.485};
  //const std::vector<float> std = {0.225, 0.224, 0.229};
  auto labels = imagenetLabels(label_path);
  auto toTensor = [](const af::array& in) { return in / 255.f; };
  std::vector<Dataset::TransformFunction> train_transforms = {
      // randomly resize shortest side of image between 256 to 480 for scale
      // invariance
      //ImageDataset::randomResizeCropTransform(224, 0.08, 1.0, 3./.4, 4./3.),
      //// Randomly flip image with probability of 0.5
      //ImageDataset::horizontalFlipTransform(0.5),
      //ImageDataset::normalizeImage(mean, std)
      ImageDataset::resizeTransform(256),
      ImageDataset::centerCrop(224),
      toTensor,
      //[](const af::array& in) { return in.as(f32) / 255.f; },
      //[](const af::array& in) { return in.as(f32) / 255.f; },
      //[](const af::array& in) { return af::constant(0.01, in.dims()); },
      //zeros
      ImageDataset::normalizeImage(mean, std)
  };
  std::vector<Dataset::TransformFunction> val_transforms = {
      // Resize shortest side to 256, then take a center crop
      ImageDataset::resizeTransform(256),
      ImageDataset::centerCrop(224),
      toTensor,
      //[](const af::array& in) { return in.as(f32) / 255.f;},
      //[](const af::array& in) { return af::constant(0.01, in.dims()); },
      //zeros
      ImageDataset::normalizeImage(mean, std)
  };
  //const uint64_t miniBatchSize = batch_size / world_size;
  auto test = std::make_shared<ImageDataset>(
          imagenetDataset(train_list, labels, train_transforms));
  //auto test = std::make_shared<NumpyDataset>(1024, "/private/home/padentomasello/tmp/pytorch_dump/save/train/");
  auto train_ds = DistributedDataset(
      test,
      world_rank,
      world_size,
      miniBatchSize,
      prefetch_threads,
      prefetch_size, true);

  //auto test_val = std::make_shared<NumpyDataset>(128, "/private/home/padentomasello/tmp/pytorch_dump/save/val/");
  auto test_val = std::make_shared<ImageDataset>(imagenetDataset(val_list, labels, val_transforms));
  auto val_ds = DistributedDataset(
      test_val,
      world_rank,
      world_size,
      miniBatchSize,
      prefetch_threads,
      prefetch_size, true);
#endif


  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  //auto model = std::make_shared<Sequential>(resnet34());
  auto model = std::make_shared<Sequential>(resnet34small());
  // synchronize parameters of tje model so that the parameters in each process
  // is the same

  //SGDOptimizer opt(model->params(), learning_rate, momentum, weight_decay);
  SGDOptimizer opt(model->params(), learning_rate);

  auto lrScheduler = [&opt, &learning_rate](int epoch) {
    // Adjust learning rate every 30 epoch
    if (epoch > 0 && epoch % 30 == 0) {
      const float newLr = opt.getLr() * 0.1;
      std::cout << "Setting learning rate to: " << newLr << std::endl;
      opt.setLr(newLr);
    }
  };

  // Small utility functions to load and save models
  std::string checkpointPrefix = "/private/home/padentomasello/code/flashlight/build/model-max";
  auto saveModel = [world_rank, &model, &checkpointPrefix](int epoch) {
    if(world_rank == 0) {
      std::string modelPath = checkpointPrefix + std::to_string(epoch);
      std::cout <<  "Saving model to file: " << modelPath << std::endl;
      fl::save(modelPath, model);
    }
  };

  auto loadModel = [&model, &checkpointPrefix](int epoch) {
      std::string modelPath = checkpointPrefix + std::to_string(epoch);
      std::cout <<  "Loading model from file: " << modelPath << std::endl;
      fl::load(modelPath, model);
  };
  int checkpointEpoch = -1;
  if (checkpointEpoch >= 0) {
    loadModel(checkpointEpoch);
  }

#if DISTRIBUTED
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);
#endif

#if 0
  for (int i = 0; i < epochs; i++) {
    int idx = 0;
    for (auto& example : train_ds) {
      auto test = example[ImageDataset::INPUT_IDX].scalar<float>();
      if (test == 0.0) {
        std::cout << "Impossible " << std::endl;
      }
      if (idx % 10) {
        std::cout << "Epoch" << i << "Idx " << idx << std::endl;
      }
      idx++;
    }
}

#else
  // The main training loop
  TimeMeter time_meter;
  TopKMeter top5_meter(5, true);
  TopKMeter top1_meter(1, true);
  AverageValueMeter train_loss_meter;
  for (int e = (checkpointEpoch + 1); e < epochs; e++) {
    train_ds.resample();
    lrScheduler(e);
    if (TRAIN) {
      model->train();
    } else {
      model->eval();
    }

    // Get an iterator over the data
    time_meter.resume();
    int idx = 0;
    for (auto& example : train_ds) {
      opt.zeroGrad();
      // Make a Variable from the input array.
      auto inputs = noGrad(example[ImageDataset::INPUT_IDX]);

      // Get the activations from the model.
      auto output = model->forward(inputs);

      // Make a Variable from the target array.
      auto target = noGrad(example[ImageDataset::TARGET_IDX]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);

      train_loss_meter.add(loss.array());
      top5_meter.add(output.array(), target.array());
      top1_meter.add(output.array(), target.array());

      // Backprop, update the weights and then zero the gradients.

#if DISTRIBUTED
      reducer->finalize();
#endif

#if TRAIN
      loss.backward();
      opt.step();
#endif

      // Compute and record the prediction error.
      if (++idx % 10 == 0) {
        double train_loss = train_loss_meter.value()[0];
        double time = time_meter.value();
        double sample_per_second = ((idx * miniBatchSize) / time);
        double top5 = top5_meter.value();
        double top1 = top1_meter.value();
        af::array train_loss_arr = af::array(1, &train_loss);
        af::array top1_arr = af::array(1, &top1);
        af::array top5_arr = af::array(1, &top5);
        af::array samples_per_second_arr = af::array(1, &sample_per_second);
        std::vector<af::array*> metric_arrays = {
          &train_loss_arr, &top1_arr, &top5_arr, &samples_per_second_arr
        };
#if DISTRIBUTED
        fl::allReduceMultiple(metric_arrays, false, false);
#endif
        if (world_rank == 0) {
          std::cout << "Epoch " << e << std::setprecision(6) << " Batch: " << idx
                    << " Samples per second " << samples_per_second_arr.scalar<double>()
                    << ": Avg Train Loss: " << train_loss_arr.scalar<double>() / world_size
                    << ": Train Top5 Error( %): " << top5_arr.scalar<double>() / world_size
                    << ": Train Top1 Error( %): " << top1_arr.scalar<double>() / world_size
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
#endif
}
