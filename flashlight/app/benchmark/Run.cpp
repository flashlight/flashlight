/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include "flashlight/app/benchmark/ModelBenchmarker.h"
#include "flashlight/app/benchmark/Utils.h"
#include "flashlight/app/benchmark/models/AsrTransformer.h"
#include "flashlight/app/benchmark/models/LmTransformer.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/lib/text/String.h"
#include "flashlight/pkg/runtime/common/DistributedUtils.h"
#include "flashlight/pkg/speech/criterion/criterion.h"
#include "flashlight/pkg/vision/criterion/SetCriterion.h"
#include "flashlight/pkg/vision/models/Detr.h"
#include "flashlight/pkg/vision/models/Resnet.h"
#include "flashlight/pkg/vision/models/Resnet50Backbone.h"
#include "flashlight/pkg/vision/models/ViT.h"

DEFINE_bool(log_verbose, false, "Log out detailed running time benchmark");

DEFINE_bool(distributed_enable, false, "Enable distributed training");
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

/* ------------------------------- ViTBase ------------------------------- */
void runViTBase(bool fp16 = false) {
  fl::app::benchmark::init();

  // Data
  const int batchsize = 64, imgSize = 224;
  auto input = fl::input(fl::rand({imgSize, imgSize, 3, batchsize}));
  auto target = fl::noGrad(fl::rand({1000, batchsize}));
  if (fp16) {
    input = input.astype(fl::dtype::f16);
    target = target.astype(fl::dtype::f16);
  }

  // Model
  std::shared_ptr<fl::Module> model = std::make_shared<fl::pkg::vision::ViT>(
      12, // FLAGS_model_layers,
      768, // FLAGS_model_hidden_emb_size,
      3072, // FLAGS_model_mlp_size,
      12, // FLAGS_model_heads,
      0., // FLAGS_train_dropout,
      0.1, // FLAGS_train_layerdrop,
      1000 // labelMap.size()
  );

  auto criterion =
      [&target](const std::vector<fl::Variable>& input) -> fl::Variable {
    auto output = logSoftmax(input.front(), 0).astype(target.type());
    auto loss = fl::mean(fl::sum(fl::negate(target * output), {0}), {1});
    return loss;
  };

  // Test
  fl::app::benchmark::ModelBenchmarker benchmarker(
      model, criterion, fl::getWorldSize());
  benchmarker.runBenchmark({input});

  // Print
  fl::app::benchmark::printInfo(
      "ViTBase", fp16, benchmarker, batchsize, FLAGS_log_verbose);
}

/* ------------------------------- ResNet34 ------------------------------- */
void runResNet34(bool fp16 = false) {
  fl::app::benchmark::init();

  // Data
  const int batchsize = 192, imgSize = 224;
  auto input = fl::input(fl::rand({imgSize, imgSize, 3, batchsize}));
  auto target = fl::noGrad(fl::rand({batchsize}) * 1000).astype(fl::dtype::s32);
  if (fp16) {
    input = input.astype(fl::dtype::f16);
  }

  // Model
  std::shared_ptr<fl::Module> model = fl::pkg::vision::resnet34();

  auto criterion =
      [&target](const std::vector<fl::Variable>& input) -> fl::Variable {
    return categoricalCrossEntropy(logSoftmax(input.front(), 0), target);
  };

  // Test
  fl::app::benchmark::ModelBenchmarker benchmarker(
      model, criterion, fl::getWorldSize());
  benchmarker.runBenchmark({input});

  // Print
  fl::app::benchmark::printInfo(
      "ResNet34", fp16, benchmarker, batchsize, FLAGS_log_verbose);
}

/* ------------------------------- ResNet50 ------------------------------- */
void runResNet50(bool fp16 = false) {
  fl::app::benchmark::init();

  // Data
  const int batchsize = 192, imgSize = 224;
  auto input = fl::input(fl::rand({imgSize, imgSize, 3, batchsize}));
  auto target = fl::noGrad(fl::rand({batchsize}) * 1000).astype(fl::dtype::s32);
  if (fp16) {
    input = input.astype(fl::dtype::f16);
  }

  // Model
  std::shared_ptr<fl::Module> model = fl::pkg::vision::resnet50();

  auto criterion =
      [&target](const std::vector<fl::Variable>& input) -> fl::Variable {
    return categoricalCrossEntropy(logSoftmax(input.front(), 0), target);
  };

  // Test
  fl::app::benchmark::ModelBenchmarker benchmarker(
      model, criterion, fl::getWorldSize());
  benchmarker.runBenchmark({input});

  // Print
  fl::app::benchmark::printInfo(
      "ResNet50", fp16, benchmarker, batchsize, FLAGS_log_verbose);
}

/* ------------------------------- Detr ------------------------------- */
void runDetr(bool fp16 = false) {
  fl::app::benchmark::init();

  // Data
  const auto dataType = fp16 ? fl::dtype::f16 : fl::dtype::f32;
  const int batchsize = 12, numObjs = 4;
  const int imgSize = 800;
  auto input = fl::input(fl::rand({imgSize, imgSize, 3, batchsize}, dataType));
  auto mask = fl::input(fl::rand({imgSize, imgSize, 1, batchsize}));

  std::vector<fl::Variable> targetBoxes(batchsize);
  std::vector<fl::Variable> targetClasses(batchsize);
  for (int i = 0; i < batchsize; i++) {
    auto boxes = fl::rand({4, numObjs}, dataType) * (imgSize - 20);
    boxes(fl::span, fl::range(2, 4)) = boxes(fl::span, fl::range(0, 2)) + 15;
    targetBoxes.push_back(fl::noGrad(boxes));
    targetClasses.push_back(fl::noGrad(fl::rand({1, numObjs}, dataType) * 91));
  }

  // Model
  auto backbone = std::make_shared<fl::pkg::vision::Resnet50Backbone>();
  auto transformer = std::make_shared<fl::pkg::vision::Transformer>(
      256, // modelDim
      8, // numHeads
      6, // numEncoderLayers
      6, // numDecoderLayers
      2048, // mlpDim,
      0.1 // pDropout
  );
  std::shared_ptr<fl::Module> model = std::make_shared<fl::pkg::vision::Detr>(
      transformer,
      backbone,
      256, // hiddenDim
      91, // numClasses
      100, // numQueries
      false // auxLoss
  );

  // Criterion
  const float setCostClass = 1.f;
  const float setCostBBox = 5.f;
  const float setCostGiou = 2.f;
  const float bboxLossCoef = 5.f;
  const float giouLossCoef = 2.f;

  auto matcher =
      fl::pkg::vision::HungarianMatcher(setCostClass, setCostBBox, setCostGiou);
  const std::unordered_map<std::string, float> lossWeightsBase = {
      {"lossCe", 1.f}, {"lossGiou", giouLossCoef}, {"lossBbox", bboxLossCoef}};

  std::unordered_map<std::string, float> lossWeights;
  for (int i = 0; i < 6; i++) {
    for (auto l : lossWeightsBase) {
      std::string key = l.first + "_" + std::to_string(i);
      lossWeights[key] = l.second;
    }
  }
  auto setCriterion =
      fl::pkg::vision::SetCriterion(91, matcher, lossWeights, 0.1);
  auto weightDict = setCriterion.getWeightDict();

  auto criterion =
      [&setCriterion, &weightDict, &targetBoxes, &targetClasses, &dataType](
          const std::vector<fl::Variable>& input) -> fl::Variable {
    auto loss =
        setCriterion.forward(input[1], input[0], targetBoxes, targetClasses);
    // TODO{fl::Tensor} - explore using a scalar Tensor
    auto accumLoss = fl::Variable(fl::full({1}, 0, dataType), true);
    for (auto losses : loss) {
      fl::Variable scaled_loss = weightDict[losses.first] * losses.second;
      accumLoss = scaled_loss + accumLoss;
    }
    return accumLoss;
  };

  // Test
  fl::app::benchmark::ModelBenchmarker benchmarker(
      model, criterion, fl::getWorldSize());
  benchmarker.runBenchmark({input, mask});

  // Print
  fl::app::benchmark::printInfo(
      "Detr", fp16, benchmarker, batchsize, FLAGS_log_verbose);
}

/* ----------------------------- LM Transformer ----------------------------- */
void runLmTransformer(bool fp16 = false) {
  fl::app::benchmark::init();

  // Data
  const int batchsize = 2048, numTokens = 150000;
  const std::vector<int> cutoff{10000, 50000, numTokens};
  auto rawInput = fl::rand({batchsize}) * cutoff[0];
  auto mask1 = fl::rand({batchsize}) < 0.2;
  auto mask2 = fl::rand({batchsize}) < 0.05;
  rawInput = rawInput + mask1 * cutoff[0] + mask2 * cutoff[1];
  auto input = fl::input(rawInput.astype(fl::dtype::s32));
  auto target = fl::noGrad(rawInput.astype(fl::dtype::s32));

  // Model
  std::shared_ptr<fl::Module> model =
      std::make_shared<fl::app::benchmark::LmTransformer>(numTokens, fp16);

  // Criterion
  auto softmax = std::make_shared<fl::AdaptiveSoftMax>(
      1024, // adsm_input_size
      cutoff);
  auto adsm = std::make_shared<fl::AdaptiveSoftMaxLoss>(
      softmax, fl::ReduceMode::SUM, 1 // padIdx
  );

  auto criterion =
      [&adsm, &target](const std::vector<fl::Variable>& input) -> fl::Variable {
    adsm->train();
    return adsm->forward(input.front(), target);
  };

  // Test
  fl::app::benchmark::ModelBenchmarker benchmarker(
      model, criterion, fl::getWorldSize());
  benchmarker.runBenchmark({input});

  // Print
  fl::app::benchmark::printInfo(
      "LM Transformer", fp16, benchmarker, batchsize, FLAGS_log_verbose);
}

/* ---------------------------- ASR Transformer ---------------------------- */
void runAsrTransformer(bool fp16 = false) {
  fl::app::benchmark::init();
  if (fp16) {
    fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
  }

  // Data
  const int batchsize = 8, numFrames = 1500, numFeatures = 80;
  const int numTarget = 30, targetLength = 100;

  auto input = fl::input(fl::rand({numFrames, 1, numFeatures, batchsize}));
  auto lengths = fl::input(fl::full({1, batchsize}, numFrames));
  auto target = fl::noGrad(fl::rand({targetLength, batchsize}) * numTarget)
                    .astype(fl::dtype::s32);

  // Model
  std::shared_ptr<fl::Module> model =
      std::make_shared<fl::app::benchmark::AsrTransformer>(
          numFeatures, numTarget);

  // Criterion
  auto ctc = std::make_shared<fl::pkg::speech::CTCLoss>(
      fl::lib::seq::CriterionScaleMode::NONE);

  auto criterion =
      [&ctc, &target](const std::vector<fl::Variable>& input) -> fl::Variable {
    return ctc->forward({input.front(), target}).front();
  };

  // Test
  fl::app::benchmark::ModelBenchmarker benchmarker(
      model, criterion, fl::getWorldSize());
  benchmarker.runBenchmark({input, lengths});

  // Print
  fl::app::benchmark::printInfo(
      "ASR Transformer",
      fp16,
      benchmarker,
      batchsize * numFrames / 100,
      FLAGS_log_verbose);
}

int main(int argc, char** argv) {
  fl::init();
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  if (FLAGS_distributed_enable) {
    fl::pkg::runtime::initDistributed(
        FLAGS_distributed_world_rank,
        FLAGS_distributed_world_size,
        FLAGS_distributed_max_devices_per_node,
        FLAGS_distributed_rndv_filepath);
  }

  runViTBase();
  runViTBase(true);

  runResNet34();
  runResNet34(true);

  runResNet50();
  runResNet50(true);

  runDetr();
  runDetr(true);

  runLmTransformer();
  runLmTransformer(true);

  runAsrTransformer();
  runAsrTransformer(true);
}
