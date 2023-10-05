/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/lm/Trainer.h"
#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/tensor/Init.h"

using namespace fl::pkg::runtime;
using namespace fl::lib;
using namespace fl::app::lm;

int main(int argc, char** argv) {
  fl::init();
  /* Parse or load persistent states */
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  if (FLAGS_distributed_enable) {
    initDistributed(
        FLAGS_distributed_world_rank,
        FLAGS_distributed_world_size,
        FLAGS_distributed_max_devices_per_node,
        FLAGS_distributed_rndv_filepath);
  }

  /* Select mode */
  std::string mode;
  if (fs::exists(
          fs::path(FLAGS_exp_rundir) / (FLAGS_exp_model_name + ".bin"))) {
    mode = "continue";
  } else if (!FLAGS_exp_init_model_path.empty()) {
    mode = "fork";
  } else {
    mode = "train";
  }

  /* Run train */
  Trainer trainer(mode);
  // flags may be overridden from the model
  // so reloading from command line again
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  trainer.runTraining();
}
