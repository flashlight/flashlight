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
#include "vision/dataset/Coco.h"
#include "vision/dataset/BoxUtils.h"
#include "vision/dataset/Transforms.h"
#include "vision/dataset/Utils.h"
#include "vision/models/Resnet.h"

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
    "/tmp/",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");
DEFINE_uint64(batch_size, 256, "Total batch size across all gpus");
DEFINE_string(checkpointpath, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(checkpoint, -1, "Load from checkpoint");


using namespace fl;
using namespace cv::dataset;

int main(int argc, char** argv) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string label_path = FLAGS_data_dir + "labels.txt";
  const std::string train_list = FLAGS_data_dir + "train";
  const std::string val_list = FLAGS_data_dir + "val";

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
  std::string coco_list = "/private/home/padentomasello/data/coco/train.lst";
  auto coco = cv::dataset::coco(coco_list, val_transforms);
  auto first = coco->get(0)[1];
  af_print(first);
  return 0;
}
