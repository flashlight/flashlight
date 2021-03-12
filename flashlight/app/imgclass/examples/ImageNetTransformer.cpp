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
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

#include "beta_distribution.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_uint64(data_batch_size, 256, "Batch size per gpus");
DEFINE_uint64(data_prefetch_thread, 10, "Batch size per gpus");

DEFINE_double(train_lr, 5e-4, "Learning rate");
DEFINE_double(train_warmup_updates, 10000, "train_warmup_updates");
DEFINE_double(train_beta1, 0.9, "Learning rate");
DEFINE_double(train_beta2, 0.999, "Learning rate");
DEFINE_double(train_wd, 5e-2, "Weight decay");
DEFINE_uint64(train_epochs, 50, "Number of epochs to train");
DEFINE_double(train_dropout, 0., "Number of epochs to train");
DEFINE_double(train_layerdrop, 0., "Number of epochs to train");
DEFINE_double(train_maxgradnorm, 1., "");

DEFINE_double(train_p_randomerase, 0.25, "");
DEFINE_double(train_p_randomeaug, 0.5, "");
DEFINE_uint64(train_n_randomeaug, 2, "");
DEFINE_double(train_p_mixup, 0., "");
DEFINE_double(train_p_cutmix, 1.0, "");
DEFINE_double(train_p_label_smoothing, 0.1, "");
DEFINE_uint64(train_n_repeatedaug, 3, "");
DEFINE_double(train_p_switchmix, 0.5, "");

DEFINE_bool(distributed_enable, true, "Enable distributed training");
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

DEFINE_string(exp_checkpoint_path, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(exp_checkpoint_epoch, -1, "Checkpoint epoch to load from");

DEFINE_int64(model_layers, 12, "model_layers");
DEFINE_int64(model_hidden_emb_size, 768, "model_hidden_size");
DEFINE_int64(model_mlp_size, 3072, "model_mlp_size");
DEFINE_int64(model_heads, 12, "model_heads");

// MIXED PRECISION OPTIONS
DEFINE_bool(
    fl_amp_use_mixed_precision,
    false,
    "[train] Use mixed precision for training - scale loss and gradients up and down "
    "by a scale factor that changes over time. If no fl optim mode is "
    "specified with --fl_optim_mode when passing this flag, automatically "
    "sets the optim mode to O1.");
DEFINE_double(
    fl_amp_scale_factor,
    4096.,
    "[train] Starting scale factor to use for loss scaling "
    " with mixed precision training");
DEFINE_uint64(
    fl_amp_scale_factor_update_interval,
    2000,
    "[train] Update interval for adjusting loss scaling in mixed precision training");
DEFINE_uint64(
    fl_amp_max_scale_factor,
    32000,
    "[train] Maximum value for the loss scale factor in mixed precision training");
DEFINE_string(
    fl_optim_mode,
    "",
    "[train] Sets the flashlight optimization mode. "
    "Optim modes can be O1, O2, or O3.");

using namespace fl;
using fl::ext::image::compose;
using fl::ext::image::ImageTransform;
using namespace fl::app::imgclass;

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

const double pi = std::acos(-1);

// Returns the average loss, top 5 error, and top 1 error
std::tuple<double, double, double> evalLoop(
    std::shared_ptr<Container> model,
    Dataset& dataset) {
  AverageValueMeter lossMeter;
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);

  // Place the model in eval mode.
  model->eval();
  for (auto& example : dataset) {
    auto inputs = noGrad(example[kImagenetInputIdx]);
    auto output = model->forward({inputs}).front();

    auto target = noGrad(example[kImagenetTargetIdx]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    lossMeter.add(loss.array().scalar<float>());
    top5Acc.add(output.array(), target.array());
    top1Acc.add(output.array(), target.array());
  }
  model->train();
  fl::ext::syncMeter(lossMeter);
  fl::ext::syncMeter(top5Acc);
  fl::ext::syncMeter(top1Acc);

  double loss = lossMeter.value()[0];
  return std::make_tuple(loss, top5Acc.value(), top1Acc.value());
};

af::array oneHot(const af::array& targets, int C) {
  float offValue = FLAGS_train_p_label_smoothing / C;
  float onValue = 1. - FLAGS_train_p_label_smoothing;

  int X = targets.elements();
  auto y = af::moddims(targets, af::dim4(1, X));
  auto A = af::range(af::dim4(C, X));
  auto B = af::tile(y, af::dim4(C));
  auto mask = A == B; // [C X]

  af::array out = af::constant(onValue, af::dim4(C, X));
  out = out * mask + offValue;

  return out;
}

bool isBadArray(const af::array& arr) {
  return af::anyTrue<bool>(af::isNaN(arr)) || af::anyTrue<bool>(af::isInf(arr));
}

int main(int argc, char** argv) {
  fl::init();
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string labelPath = lib::pathsConcat(FLAGS_data_dir, "labels.txt");
  const std::string trainList = lib::pathsConcat(FLAGS_data_dir, "train");
  const std::string valList = lib::pathsConcat(FLAGS_data_dir, "val");

  /////////////////////////
  // Setup distributed training
  ////////////////////////
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
  const bool isMaster = (worldRank == 0);

  // af::setSeed(worldRank * 4399);
  // af::setSeed(worldSize);
  af::setSeed(worldRank);
  std::srand(worldRank * 4399);

  auto reducer =
      std::make_shared<fl::CoalescingReducer>(1.0 / worldSize, true, true);

  //////////////////////////
  //  Create datasets
  /////////////////////////
  FL_LOG_MASTER(INFO) << "Creating dataset";
  // These are the mean and std for each channel of Imagenet
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const int randomResizeMax = 480;
  const int randomResizeMin = 256;
  const int randomCropSize = 224;
  const float horizontalFlipProb = 0.5f;
  // TransformDataset will apply each transform in a vector to the respective
  // af::array. Thus, we need to `compose` all of the transforms so are each
  // applied only to the image
  ImageTransform trainTransforms = compose({
      // randomly resize shortest side of image between 256 to 480 for
      // scale invariance
      // fl::ext::image::c(randomResizeMin,
      // randomResizeMax),
      // fl::ext::image::randomCropTransform(randomCropSize, randomCropSize),
      fl::ext::image::randomResizeCropTransform(
          randomCropSize, 0.08, 1.0, 3. / 4., 4. / 3.),
      fl::ext::image::randomHorizontalFlipTransform(horizontalFlipProb),
      fl::ext::image::randomAugmentationTransform(
          FLAGS_train_p_randomeaug, FLAGS_train_n_randomeaug),
      fl::ext::image::normalizeImage(mean, std),
      fl::ext::image::randomEraseTransform(FLAGS_train_p_randomerase)
      // end
  });
  ImageTransform valTransforms =
      compose({// Resize shortest side to 256, then take a center crop
               fl::ext::image::resizeTransform(randomResizeMin),
               fl::ext::image::centerCropTransform(randomCropSize),
               fl::ext::image::normalizeImage(mean, std)});

  const int64_t batchSizePerGpu = FLAGS_data_batch_size;
  const int64_t prefetchSize = FLAGS_data_batch_size * 10;
  auto labelMap = getImagenetLabels(labelPath);
  auto trainDataset = std::make_shared<fl::ext::image::DistributedDataset>(
      imagenetDataset(trainList, labelMap, {trainTransforms}),
      // imagenetDataset(trainList, labelMap, {valTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      FLAGS_train_n_repeatedaug,
      FLAGS_data_prefetch_thread,
      prefetchSize,
      fl::BatchDatasetPolicy::SKIP_LAST);
  FL_LOG_MASTER(INFO) << "[trainDataset size] " << trainDataset->size();
  // auto trainDataset1 = std::make_shared<fl::ext::image::DistributedDataset>(
  //     imagenetDataset(trainList, labelMap, {trainTransforms}),
  //     worldRank,
  //     worldSize,
  //     batchSizePerGpu,
  //     prefetchThreads,
  //     prefetchSize,
  //     fl::BatchDatasetPolicy::SKIP_LAST);

  auto valDataset = fl::ext::image::DistributedDataset(
      imagenetDataset(valList, labelMap, {valTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      1,
      FLAGS_data_prefetch_thread,
      prefetchSize,
      fl::BatchDatasetPolicy::INCLUDE_LAST);
  FL_LOG_MASTER(INFO) << "[valDataset size] " << valDataset.size();

  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  auto model = std::make_shared<fl::ext::image::ViT>(
      FLAGS_model_layers,
      FLAGS_model_hidden_emb_size,
      FLAGS_model_mlp_size,
      FLAGS_model_heads,
      FLAGS_train_dropout,
      FLAGS_train_layerdrop,
      1000);
  // auto model = std::make_shared<fl::ext::image::ViT>(
  //     "/private/home/qiantong/tmp/vitb_pt_init/");
  FL_LOG_MASTER(INFO) << "[model with parameters " << fl::numTotalParams(model)
                      << "] " << model->prettyString();
  // synchronize parameters of the model so that the parameters in each process
  // is the same
  fl::allReduceParameters(model);
  fl::distributeModuleGrads(model, reducer);

  auto opt = AdamOptimizer(
      model->params(),
      FLAGS_train_lr,
      FLAGS_train_beta1,
      FLAGS_train_beta2,
      1e-8,
      FLAGS_train_wd);

  // auto opt =
  //     fl::SGDOptimizer(model->params(), FLAGS_train_lr, 0, FLAGS_train_wd);

  // auto lrScheduler = [&opt](int epoch, int update) {
  //   double lr = FLAGS_train_lr;
  //   if (update < FLAGS_train_warmup_updates) {
  //     // warmup stage
  //     lr = 1e-7 +
  //         (FLAGS_train_lr - 1e-7) * update /
  //             (double(FLAGS_train_warmup_updates));
  //   } else {
  //     // cosine decay
  //     lr = 1e-6 +
  //         0.5 * FLAGS_train_lr *
  //             (std::cos(((double)epoch) / ((double)FLAGS_train_epochs) * pi)
  //             +
  //              1);
  //   }
  //   opt.setLr(lr);
  // };

  auto lrScheduler = [&opt](int epoch) {
    double lr = FLAGS_train_lr;
    if (epoch <= 5) {
      // warmup stage
      lr = (epoch - 1) * FLAGS_train_lr / 5;
      lr = std::max(lr, 1e-6);
    } else {
      // cosine decay
      lr = 1e-5 +
          0.5 * (FLAGS_train_lr - 1e-5) *
              (std::cos(((double)epoch - 1) / ((double)FLAGS_train_epochs) * pi) +
               1);
    }
    opt.setLr(lr);
  };

  // Small utility functions to load and save models
  int batchIdx = 0;
  int epoch = 0;

  auto saveModel =
      [&model, &isMaster, &epoch, &batchIdx](const std::string& suffix = "") {
        if (isMaster) {
          std::string modelPath = FLAGS_exp_checkpoint_path + suffix;
          FL_LOG_MASTER(INFO) << "Saving model to file: " << modelPath;
          fl::save(modelPath, model, batchIdx, epoch);
        }
      };

  auto loadModel = [&model, &epoch, &batchIdx]() {
    FL_LOG_MASTER(INFO) << "Loading model from file: "
                        << FLAGS_exp_checkpoint_path;
    fl::load(FLAGS_exp_checkpoint_path, model, batchIdx, epoch);
  };
  if (FLAGS_exp_checkpoint_epoch >= 0) {
    loadModel();
  }

  auto betaGenerator = sftrabbit::beta_distribution<float>(0.8, 0.8);

  //////////////////////////
  // The main training loop
  /////////////////////////
  if (FLAGS_fl_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    FL_LOG_MASTER(INFO)
        << "Mixed precision training enabled. Will perform loss scaling.";
    if (FLAGS_fl_optim_mode.empty()) {
      // fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::DEFAULT);
    }
  }
  unsigned short scaleCounter = 1;
  double scaleFactor =
      FLAGS_fl_amp_use_mixed_precision ? FLAGS_fl_amp_scale_factor : 1.;
  unsigned int kScaleFactorUpdateInterval =
      FLAGS_fl_amp_scale_factor_update_interval;
  unsigned int kMaxScaleFactor = FLAGS_fl_amp_max_scale_factor;

  fl::TimeMeter sampleTimerMeter{true};
  fl::TimeMeter fwdTimeMeter{true};
  fl::TimeMeter critFwdTimeMeter{true};
  fl::TimeMeter bwdTimeMeter{true};
  fl::TimeMeter optimTimeMeter{true};

  FL_LOG_MASTER(INFO) << "[Training starts]";
  TimeMeter timeMeter;
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);
  AverageValueMeter trainLossMeter;
  for (; epoch < FLAGS_train_epochs; epoch++) {
    // std::cout << "---" << std::endl;
    trainDataset->resample(epoch);
    // trainDataset1->resample(epoch + 4399);
    std::mt19937_64 engine(epoch);

    // Get an iterator over the data
    timeMeter.resume();
    for (int idx = 0; idx < trainDataset->size(); idx++, batchIdx++) {
      // std::cout << fl::getWorldRank() << ": " << scaleFactor << std::endl;
      Variable loss;
      sampleTimerMeter.resume();

#if 0
      af::array input = trainDataset->get(idx)[kImagenetInputIdx];
      input = trainTransforms(input);
      for (int i = 1;
           i < std::min((int)FLAGS_data_batch_size, (int)trainDataset->size());
           i++) {
        auto tmp = trainDataset->get(idx + i)[kImagenetInputIdx];
        tmp = trainTransforms(tmp);
        input = af::join(3, input, tmp);
      }

      // af::array input = af::randu(224, 224, 3);
      // for (int i = 1; i < FLAGS_data_batch_size; i++) {
      //   auto tmp = af::randu(224, 224, 3);
      //   tmp = trainTransforms(tmp);
      //   input = af::join(3, input, tmp);
      // }

      auto inputs = fl::Variable(input, false);
      auto y = fl::Variable(af::randu(1000, FLAGS_data_batch_size), false);
      // if (FLAGS_fl_amp_use_mixed_precision) {
      //   inputs = inputs.as(f16);
      // }
#endif
#if 1
      auto sample = trainDataset->get(idx);
      // auto sample1 = trainDataset1->get(idx);

      auto rawInput = sample[kImagenetInputIdx];
      // if (FLAGS_fl_amp_use_mixed_precision) {
      //   rawInput = rawInput.as(f16);
      // }
      auto rawInput1 = af::flip(rawInput, 3);
      // auto rawInput1 = sample1[kImagenetInputIdx];
      // if (FLAGS_fl_amp_use_mixed_precision) {
      //   rawInput1 = rawInput.as(f16);
      // }
      // rawInput1 = af::flip(rawInput1, 3);
      rawInput1.eval();

      auto rawTarget = sample[kImagenetTargetIdx];
      // af_print(af::transpose(rawTarget));
      // if (idx == 10) exit(0);
      auto rawTarget1 = af::flip(rawTarget, 0);
      // af_print(af::transpose(rawTarget1));
      // auto rawTarget1 = sample1[kImagenetTargetIdx];
      // rawTarget1 = af::flip(rawTarget1, 0);
      auto target = oneHot(rawTarget, 1000);
      auto target1 = oneHot(rawTarget1, 1000);
      rawTarget1.eval();

      lrScheduler(epoch);
      opt.zeroGrad();

      // Make a Variable from the input array.
      Variable inputs = noGrad(rawInput);
      float mixP =
          static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
      float example1Weight = 0.;
      if (FLAGS_train_p_mixup > 0 && mixP > FLAGS_train_p_switchmix) {
        // mixup
        // float lambda = 0.9 +
        //     0.1 * static_cast<float>(std::rand()) /
        //         static_cast<float>(RAND_MAX);
        float lambda = betaGenerator(engine);
        // std::cout << lambda << std::endl;
        inputs = noGrad(lambda * rawInput1 + (1 - lambda) * rawInput);
        example1Weight = lambda;
      } else if (FLAGS_train_p_cutmix > 0 && mixP <= FLAGS_train_p_switchmix) {
        // cut mix
        float lambda =
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        float lambdaSqrt = std::sqrt(lambda);
        const int w = rawInput.dims(0);
        const int h = rawInput.dims(1);
        const int maskW = std::round(w * lambdaSqrt);
        const int maskH = std::round(h * lambdaSqrt);
        // if (maskW == 0 || maskH == 0) {
        //   inputs = noGrad(rawInput);
        //   example1Weight = 0.;
        // } else if (maskW == w || maskH == h) {
        //   inputs = noGrad(rawInput1);
        //   example1Weight = 1.;
        // } else {
        //   const int x = std::rand() % (w - maskW);
        //   const int y = std::rand() % (h - maskH);
        //   // std::cout << x << " " << y << ", " << maskW << std::endl;

        //   rawInput(
        //       af::seq(x, x + maskW - 1),
        //       af::seq(y, y + maskH - 1),
        //       af::span,
        //       af::span) =
        //       rawInput1(
        //           af::seq(x, x + maskW - 1),
        //           af::seq(y, y + maskH - 1),
        //           af::span,
        //           af::span);
        //   inputs = noGrad(rawInput);
        //   example1Weight = lambda;
        // }

        const int centerW =
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * w;
        const int centerH =
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * h;

        const int x1 = std::max(0, centerW - maskW / 2);
        const int x2 = std::min(w - 1, centerW + maskW / 2);
        const int y1 = std::max(0, centerH - maskH / 2);
        const int y2 = std::min(h - 1, centerH + maskH / 2);
        rawInput(af::seq(x1, x2), af::seq(y1, y2), af::span, af::span) =
            rawInput1(af::seq(x1, x2), af::seq(y1, y2), af::span, af::span);
        inputs = noGrad(rawInput);
        example1Weight = (float)(x2 - x1) * (y2 - y1) / (w * h);
        // std::cout << example1Weight << std::endl;
        // } else {
        //   throw std::runtime_error("wtf no mix is used");
      }
      inputs.eval();

      auto y = noGrad(example1Weight * target1 + (1 - example1Weight) * target);
      rawInput1 = af::array();
#endif
      af::sync();
      sampleTimerMeter.stopAndIncUnit();
      // if (idx == 0) {
      //   af_print(rawTarget(af::seq(0, 9)));
      // }

      bool retrySample = false;
      do {
        retrySample = false;
        // Get the activations from the model.
        fwdTimeMeter.resume();
        auto output = model->forward({inputs}).front();

        // Make a Variable from the target array.
        // Compute and record the loss + label smoothing
        critFwdTimeMeter.resume();

        y = y.as(output.type());
        // std::cout << inputs.type() << output.type() << y.type() << std::endl;

        loss = fl::mean(fl::sum(fl::negate(y * output), {0}), {1});

        if (FLAGS_fl_amp_use_mixed_precision) {
          ++scaleCounter;
          loss = loss * scaleFactor;
        }

        // if (isBadArray(loss.array())) {
        //   FL_LOG(FATAL) << "Loss has NaN values, in proc: "
        //                 << fl::getWorldRank();
        // }
        af::sync();
        fwdTimeMeter.stopAndIncUnit();
        critFwdTimeMeter.stopAndIncUnit();

        // Backprop, update the weights and then zero the gradients.
        bwdTimeMeter.resume();
        loss.backward();
        if (FLAGS_distributed_enable) {
          reducer->finalize();
        }
        af::sync();
        bwdTimeMeter.stopAndIncUnit();

        optimTimeMeter.resume();
        if (FLAGS_fl_amp_use_mixed_precision) {
          for (auto& p : model->params()) {
            p.grad() = p.grad() / scaleFactor;
            if (isBadArray(p.grad().array())) {
              FL_LOG(INFO) << "Grad has NaN values in 3, in proc: "
                           << fl::getWorldRank();
              if (scaleFactor >= fl::kAmpMinimumScaleFactorValue) {
                scaleFactor = scaleFactor / 2.0f;
                FL_LOG(INFO)
                    << "AMP: Scale factor decreased (grad). New value:\t"
                    << scaleFactor;
                retrySample = true;
              } else {
                FL_LOG(FATAL)
                    << "Minimum loss scale reached: "
                    << fl::kAmpMinimumScaleFactorValue
                    << " with over/underflowing gradients. Lowering the "
                    << "learning rate, using gradient clipping, or "
                    << "increasing the batch size can help resolve "
                    << "loss explosion.";
              }
              scaleCounter = 1;
              break;
            }
          }
        }
        if (retrySample) {
          optimTimeMeter.stop();
          opt.zeroGrad();
          continue;
        }

        trainLossMeter.add(loss.array() / scaleFactor);
      } while (retrySample);

      // clamp gradients
      // if (FLAGS_train_maxgradnorm > 0) {
      //   fl::clipGradNorm(model->params(), FLAGS_train_maxgradnorm);
      // }
      opt.step();
      optimTimeMeter.stopAndIncUnit();

      // Compute and record the prediction error.
      int interval = 100;
      if (idx && idx % interval == 0) {
        timeMeter.stop();
        fl::ext::syncMeter(trainLossMeter);
        fl::ext::syncMeter(timeMeter);
        fl::ext::syncMeter(top5Acc);
        fl::ext::syncMeter(top1Acc);
        double time = timeMeter.value() / interval;
        double samplePerSecond = FLAGS_data_batch_size * worldSize / time;
        FL_LOG_MASTER(INFO)
            << "Epoch " << epoch << std::setprecision(5) << " Batch: " << idx
            << " Throughput " << samplePerSecond
            << " | : Batch time(ms): " << fl::lib::format("%.2f", time * 1000)
            << " : Sample Time(ms): "
            << fl::lib::format("%.2f", sampleTimerMeter.value() * 1000)
            << " : Forward Time(ms): "
            << fl::lib::format("%.2f", fwdTimeMeter.value() * 1000)
            << " : Criterion Forward Time(ms): "
            << fl::lib::format("%.2f", critFwdTimeMeter.value() * 1000)
            << " : Backward Time(ms): "
            << fl::lib::format("%.2f", bwdTimeMeter.value() * 1000)
            << " : Optimization Time(ms): "
            << fl::lib::format("%.2f", optimTimeMeter.value() * 1000)
            << " | LR: " << opt.getLr()
            << ": Avg Train Loss: " << trainLossMeter.value()[0]
            << ": Train Top5 Accuracy( %): " << top5Acc.value()
            << ": Train Top1 Accuracy( %): " << top1Acc.value();
        top5Acc.reset();
        top1Acc.reset();
        trainLossMeter.reset();
        timeMeter.reset();
        timeMeter.resume();
      }
    }
    timeMeter.stop();
    timeMeter.reset();
    sampleTimerMeter.reset();
    fwdTimeMeter.reset();
    critFwdTimeMeter.reset();
    bwdTimeMeter.reset();
    optimTimeMeter.reset();

    double valLoss, valTop1Error, valTop5Err;
    std::tie(valLoss, valTop5Err, valTop1Error) = evalLoop(model, valDataset);

    FL_LOG_MASTER(INFO) << "Epoch " << epoch << std::setprecision(5)
                        << " Validation Loss: " << valLoss
                        << " Validation Top5 Error (%): " << valTop5Err
                        << " Validation Top1 Error (%): " << valTop1Error;
    saveModel();
  }
  FL_LOG_MASTER(INFO) << "Training complete";

  return 0;
}
