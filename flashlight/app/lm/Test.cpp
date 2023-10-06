/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/lm/Trainer.h"
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

  /* Run Evaluation */
  Trainer trainer("eval");
  FL_LOG_MASTER(INFO) << "Running evaluation with model: "
                      << FLAGS_exp_init_model_path << ", on dataset: "
                      << fs::path(FLAGS_data_dir) / FLAGS_data_valid;

  auto loss = trainer.runEvaluation();
  FL_LOG_MASTER(INFO) << "Valid Loss: " << format("%.2f", loss)
                      << ", Valid PPL: " << format("%.2f", std::exp(loss));
}
