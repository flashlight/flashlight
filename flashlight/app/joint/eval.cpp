/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/joint/Trainer.h"
#include "flashlight/app/joint/Flags.h"

using namespace fl::ext;
using namespace fl::lib;
using namespace fl::app::asr;
using namespace fl::app::joint;
using namespace fl::app::lm;

int main(int argc, char** argv) {
  /* Parse or load persistent states */
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_distributed_enable) {
    initDistributed(
        FLAGS_distributed_world_rank,
        FLAGS_distributed_world_size,
        FLAGS_distributed_max_devices_per_node,
        FLAGS_distributed_rndv_filepath);
  }

  /* Run train */
  Trainer trainer("eval");
  // flags may be overridden from the model
  // so reloading from command line again
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  // trainer.evalLM();
}
