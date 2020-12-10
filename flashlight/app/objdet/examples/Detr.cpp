/**
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <iomanip>

#include <gflags/gflags.h>

#include "flashlight/app/objdet/criterion/SetCriterion.h"
#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "flashlight/app/objdet/dataset/Coco.h"
#include "flashlight/app/objdet/nn/PositionalEmbeddingSine.h"
#include "flashlight/app/objdet/nn/Transformer.h"
#include "flashlight/app/objdet/nn/Detr.h"

#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/image/af/Transforms.h"
//#include "flashlight/ext/image/fl/models/Resnet50Backbone.h"
#include "flashlight/ext/image/fl/models/Resnet50Backbone.h"
#include "flashlight/ext/image/fl/models/Resnet.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"

DEFINE_string(data_dir, "/private/home/padentomasello/data/coco3/", "Directory of imagenet data");
DEFINE_double(lr, 0.0001f, "Learning rate");
DEFINE_double(momentum, 0.9f, "Momentum");
DEFINE_uint64(metric_iters, 5, "Print metric every");

DEFINE_double(wd, 1e-4f, "Weight decay");
DEFINE_uint64(epochs, 50, "Epochs");
DEFINE_uint64(eval_iters, 1, "Epochs");
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
DEFINE_bool(enable_distributed, true, "Enable distributed training");
DEFINE_uint64(batch_size, 16, "Total batch size across all gpus");
DEFINE_string(checkpointpath, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(checkpoint, -1, "Load from checkpoint");

DEFINE_string(eval_dir, "/private/home/padentomasello/data/coco/output/", "Directory to dump images to run evaluation script on");
DEFINE_bool(print_params, false, "Directory to dump images to run evaluation script on");
DEFINE_bool(pretrained, true, "Directory to dump images to run evaluation script on");


using namespace fl;
using namespace fl::ext::image;
using namespace fl::app::objdet;

// TODO Refactor
//const int32_t backboneChannels = 512;


void printParamsAndGrads(std::shared_ptr<fl::Module> mod) {
  auto params = mod->params();
  int i = 0;
  for(auto param : params) {
    double paramMean = af::mean<double>(param.array());
    double paramStd = af::stdev<double>(param.array());
    double gradMean = -1.111111111111;
    double gradStd = -1.111111111111;
    if(param.isGradAvailable()) {
      auto grad = param.grad();
      gradMean = af::mean<double>(grad.array());
      gradStd = af::stdev<double>(grad.array());
    }
    std::cout << " i: " << i
      << " mean: " << paramMean
      << " std: " << paramStd
      << " grad mean: " << gradMean
      << " grad std: " << gradStd
      << std::endl;
    i++;
  }
}

int main(int argc, char** argv) {
  std::stringstream ss;
  ss << "PYTHONPATH=/private/home/padentomasello/code/detection-transformer/ "
    //<< "LD_LIBRARY_PATH=/private/home/padentomasello/usr/lib/:$LD_LIBRARY_PATH "
    << "/private/home/padentomasello/.conda/envs/coco/bin/python3.8 "
    << "-c 'import arrayfire as af'";
  system(ss.str().c_str());

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  //const std::string label_path = FLAGS_data_dir + "labels.txt";
  //const std::string train_list = FLAGS_data_dir + "train";
  //const std::string val_list = FLAGS_data_dir + "val";

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  if (FLAGS_enable_distributed) {
    fl::distributedInit(
    fl::DistributedInit::FILE_SYSTEM,
    FLAGS_world_rank,
    FLAGS_world_size,
    {{fl::DistributedConstants::kMaxDevicePerNode,
      std::to_string(8)},
     {fl::DistributedConstants::kFilePath, FLAGS_rndv_filepath}});
  }
  af::info();
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();

  //af::setDevice(worldRank);
  //af::setSeed(worldSize);
  std::cout << "World rank: " << worldRank << std::endl;

  auto reducer = std::make_shared<fl::CoalescingReducer>(
      1.0 / worldSize,
      true,
      true);

  /////////////////////////
  // Setup distributed training
  ////////////////////////

  //////////////////////////
  //  Create datasets
  /////////////////////////
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  std::vector<ImageTransform> train_transforms = {
      // randomly resize shortest side of image between 256 to 480 for scale 
      // invariance
      //randomResizeTransform(256, 480),
      //randomCropTransform(224, 224), normalizeImage(mean, std),
      // Randomly flip image with probability of 0.5
      //horizontalFlipTransform(0.5)
  };
  std::vector<ImageTransform> val_transforms = {
      // Resize shortest side to 256, then take a center crop
      //resizeTransform(256),
      //centerCropTransform(224),
      normalizeImage(mean, std)
  };

  const int32_t modelDim = 256;
  const int32_t numHeads = 8;
  const int32_t numEncoderLayers = 6;
  const int32_t numDecoderLayers = 6;
  const int32_t mlpDim = 2048;
  // TODO check this is correct
  const int32_t hiddenDim = modelDim;
  const int32_t numClasses = 91;
  const int32_t numQueries = 100;
  const float pDropout = 0.1;
  const bool auxLoss = false;
  std::shared_ptr<Resnet50Backbone> backbone;
  if(FLAGS_pretrained) {
    std::string modelPath = "/checkpoint/padentomasello/models/resnet50/from_pytorch_fbn";
    fl::load(modelPath, backbone);
  } else {
    backbone = std::make_shared<Resnet50Backbone>();
  }
  backbone->train();

  //
  //backbone->train();
  auto transformer = std::make_shared<Transformer>(
      modelDim,
      numHeads,
      numEncoderLayers,
      numDecoderLayers,
      mlpDim,
      pDropout);

  auto detr = std::make_shared<Detr>(
      transformer,
      backbone,
      hiddenDim,
      numClasses,
      numQueries,
      auxLoss);

  // Trained
  //std::string modelPath = "/checkpoint/padentomasello/models/detr/from_pytorch";
  // untrained but initializaed
  //std::string modelPath = "/checkpoint/padentomasello/models/detr/pytorch_initializaition";
  //fl::load(modelPath, detr);

  detr->train();
  //freezeBatchNorm(backbone);

  // synchronize parameters of tje model so that the parameters in each process
  // is the same
  fl::allReduceParameters(detr);
  //fl::allReduceParameters(backbone);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(detr, reducer);
  fl::distributeModuleGrads(backbone, reducer);

  auto saveOutput = [](
      af::array imageSizes,
      af::array imageIds,
      af::array boxes,
      af::array scores,
      std::string outputFile) {
      af::saveArray("imageSizes", imageSizes, outputFile.c_str(), false);
      af::saveArray("imageIds", imageIds, outputFile.c_str(), true);
      af::saveArray("scores", scores, outputFile.c_str(), true);
      af::saveArray("bboxes", boxes, outputFile.c_str(), true);
  };

  const float setCostClass = 1.f;
  const float setCostBBox = 5.f;
  const float setCostGiou = 2.f;
  const float bboxLossCoef = 5.f;
  const float giouLossCoef = 2.f;


  auto matcher = HungarianMatcher(
      setCostClass,
      setCostBBox,
      setCostGiou
      );
  SetCriterion::LossDict losses;


  std::unordered_map<std::string, float> lossWeightsBase = 
        { { "loss_ce" , 1.f} ,
        { "loss_giou", giouLossCoef },
        { "loss_bbox", bboxLossCoef }
  };

  std::unordered_map<std::string, float> lossWeights;
  for(int i = 0; i < numDecoderLayers; i++) {
    for(auto l : lossWeightsBase) {
      std::string key = l.first + "_" + std::to_string(i);
      lossWeights[key] = l.second;
    }
  }
  auto criterion = SetCriterion(
      numClasses,
      matcher,
      lossWeights,
      0.1,
      losses);

  auto eval_loop = [saveOutput](
      std::shared_ptr<Module> backbone,
      std::shared_ptr<Detr> model,
      std::shared_ptr<CocoDataset> dataset) {
    backbone->eval();
    model->eval();
    int idx = 0;
    for(auto& sample : *dataset) {
      std::vector<Variable> input =  { 
        fl::Variable(sample.images, false),  
        fl::Variable(sample.masks, false) 
      };
      //auto features = backbone->forward(images)[0];
      //auto features = input;
      auto output = model->forward(input);
      std::stringstream ss;
      ss << FLAGS_eval_dir << "detection" << idx << ".array";
      auto output_array = ss.str();
      int lastLayerIdx = output[0].dims(3) - 1;
      auto output_first_last = output[0].array()(af::span, af::span, af::span, af::seq(lastLayerIdx, lastLayerIdx));
      auto output_second_last = output[1].array()(af::span, af::span, af::span, af::seq(lastLayerIdx, lastLayerIdx));
      //saveOutput(sample.imageSizes, sample.imageIds, output[1].array(), output[0].array(), ss.str());
      saveOutput(sample.imageSizes, sample.imageIds, output_second_last, output_first_last, ss.str());
      idx++;
    }
    std::stringstream ss;
    ss << "PYTHONPATH=/private/home/padentomasello/code/detection-transformer/ "
      //<< "LD_LIBRARY_PATH=/private/home/padentomasello/usr/lib/:$LD_LIBRARY_PATH "
      << "/private/home/padentomasello/.conda/envs/coco/bin/python3.8 "
      << "/private/home/padentomasello/code/flashlight/flashlight/app/objdet/scripts/eval_coco.py --dir "
      << FLAGS_eval_dir;
    system(ss.str().c_str());
    std::stringstream ss2;
    ss2 << "rm -rf " << FLAGS_eval_dir << "/detection*";
    system(ss2.str().c_str());
    model->train();
    backbone->train();
    //freezeBatchNorm(backbone);
  };

  //const int64_t batch_size_per_gpu = FLAGS_batch_size / FLAGS_world_size;
  const int64_t batch_size_per_gpu = FLAGS_batch_size;
  const int64_t prefetch_threads = 10;
  const int64_t prefetch_size = FLAGS_batch_size;
  std::string coco_dir = FLAGS_data_dir;
  //std::string coco_list = "/private/home/padentomasello/data/coco-mini/train.lst";
  //auto coco = cv::dataset::coco(coco_list, val_transforms, FLAGS_batch_size);

  auto train_ds = std::make_shared<CocoDataset>(
      coco_dir + "train.lst",
      val_transforms,
      worldRank,
      worldSize,
      batch_size_per_gpu,
      prefetch_threads,
      batch_size_per_gpu, false);

  auto val_ds = std::make_shared<CocoDataset>(
      coco_dir + "val.lst",
      val_transforms,
      worldRank,
      worldSize,
      batch_size_per_gpu,
      prefetch_threads,
      batch_size_per_gpu,
      true);
  //SGDOptimizer opt(detr.params(), FLAGS_lr, FLAGS_momentum, FLAGS_wd);
  AdamOptimizer opt(detr->params(), FLAGS_lr, FLAGS_wd);
  AdamOptimizer opt2(backbone->params(), FLAGS_lr * 0.1, FLAGS_wd);
  //AdamOptimizer backbone_opt(backbone->params(), FLAGS_lr * 0.1);

  // Small utility functions to load and save models
  //auto saveModel = [&detr](int epoch) {
    //if(worldRank == 0) {
      //std::string modelPath = FLAGS_checkpointpath + std::to_string(epoch);
      //std::cout <<  "Saving model to file: " << modelPath << std::endl;
      //fl::save(modelPath, detr);
    //}
  //};

  //auto loadModel = [&detr](int epoch) {
      //std::string modelPath = FLAGS_checkpointpath + std::to_string(epoch);
      //std::cout <<  "Loading model from file: " << modelPath << std::endl;
      //fl::load(modelPath, detr);
  //};
  //if (FLAGS_checkpoint >= 0) {
    //loadModel(FLAGS_checkpoint);
  //}

  auto weightDict = criterion.getWeightDict();
  for(int e = 0; e < FLAGS_epochs; e++) {

    std::map<std::string, AverageValueMeter> meters;
    std::map<std::string, TimeMeter> timers;
    int idx = 0;
    timers["total"].resume();
    train_ds->resample();
    //while(true) {
    for(auto& sample : *train_ds) {
      //auto images =  { fl::Variable(sample.images, true) };
      std::vector<Variable> input =  { 
        fl::Variable(sample.images, false),
        fl::Variable(sample.masks, false) 
      };
      //auto features = backbone->forward(images)[0];
      ////auto features = input;
      auto output = detr->forward(input);


      ////saveOutput(sample.imageSizes, sample.imageIds, sample.target_boxes[0], sample.target_labels[0], 
          ////"/private/home/padentomasello/data/coco/scratch/labels.array");
      ////return 0;

      //timers["forward"].resume();
      ////fl::Variable features = backbone->forward(images)[0];
      //fl::Variable features = backbone->forward(images)[1];

      //fl::Variable masks = fl::Variable(
          //af::resize(
            //sample.masks, 
            //features.dims(0), 
            //features.dims(1), 
            //AF_INTERP_NEAREST),
        //true
      //);
      ////auto features = input;
      //auto output = detr->forward({features, masks});
      timers["forward"].stop();

      /////////////////////////
      // Criterion
      /////////////////////////
      std::vector<Variable> targetBoxes(sample.target_boxes.size());
      std::vector<Variable> targetClasses(sample.target_labels.size());

      std::transform(
          sample.target_boxes.begin(), sample.target_boxes.end(),
          targetBoxes.begin(),
          [](const af::array& in) { return fl::Variable(in, false); });

      std::transform(
          sample.target_labels.begin(), sample.target_labels.end(),
          targetClasses.begin(),
          [](const af::array& in) { return fl::Variable(in, false); });

      timers["criterion"].resume();

      // TODO test
      //std::vector<Variable> outputSecond = { output[1], output[1] };
      //std::vector<Variable> outputFirst = { output[0], output[0] };
      //output[0] = concatenate(outputFirst, 3);
      //output[1] = concatenate(outputSecond, 4);

      auto loss = criterion.forward(
          output[1],
          output[0],
          targetBoxes,
          targetClasses);
      auto accumLoss = fl::Variable(af::constant(0, 1), true);
      for(auto losses : loss) {
        fl::Variable scaled_loss = weightDict[losses.first] * losses.second;
        meters[losses.first].add(losses.second.array());
        meters[losses.first + "_weighted"].add(scaled_loss.array());
        accumLoss = scaled_loss + accumLoss;
      }
      meters["sum"].add(accumLoss.array());
      timers["criterion"].stop();

      /////////////////////////
      // Backward and update gradients
      //////////////////////////
      timers["backward"].resume();
      accumLoss.backward();
      timers["backward"].stop();

      if (FLAGS_enable_distributed) {
        reducer->finalize();
      }

      if(FLAGS_print_params) {
        std::cout << "Print detr params + grads" << std::endl;
        printParamsAndGrads(detr);
        std::cout << "Print backbone params + grads" << std::endl;
        printParamsAndGrads(backbone);
      }
      fl::clipGradNorm(detr->params(), 0.1);

      //freezeBatchNorm(backbone);

      opt.step();
      //opt2.step();

      opt.zeroGrad();
      //opt2.zeroGrad();
      //////////////////////////
      // Metrics
      /////////////////////////
      if(++idx % FLAGS_metric_iters == 0) {
        double total_time = timers["total"].value();
        double sample_per_second = (idx * FLAGS_batch_size * worldSize) / total_time;
        double forward_time = timers["forward"].value();
        double backward_time = timers["backward"].value();
        double criterion_time = timers["criterion"].value();
        std::cout << "Epoch: " << e << std::setprecision(5) << " | Batch: " << idx
            << " | total_time: " << total_time
            << " | idx: " << idx
            << " | sample_per_second: " << sample_per_second
            << " | forward_time_avg: " << forward_time / idx
            << " | backward_time_avg: " << backward_time / idx
            << " | criterion_time_avg: " << criterion_time / idx;
        for(auto meter : meters) {
          std::cout << " | " << meter.first << ": " << meter.second.value()[0];
        }
        std::cout << std::endl;
      }
    }
    for(auto timer : timers) {
      timer.second.reset();
    }
    for(auto meter : meters) {
      meter.second.reset();
    }
      if(e % FLAGS_eval_iters == 0 && e > 0) {
        eval_loop(backbone, detr, val_ds);
        //eval_loop(detr, val_ds);
        //saveModel(e);
      }
  }
}
