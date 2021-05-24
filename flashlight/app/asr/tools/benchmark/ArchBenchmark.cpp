/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/fl/common/SequentialBuilder.h"
#include "flashlight/pkg/runtime/plugin/ModulePlugin.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/common/String.h"

DEFINE_int32(num_iters, 50, "Number of iterations to run for benchmarking");
DEFINE_int32(batchsize, 1, "Batchsize of the input");
DEFINE_int32(in_features, 80, "Number of input features");
DEFINE_int32(out_channels, 28, "Number of output channels");
DEFINE_string(arch, "", "path to architecture file");
DEFINE_bool(use_amp, false, "Use AMP");

namespace {

float run(
    std::shared_ptr<fl::Module> network,
    fl::Variable input,
    af::array duration,
    bool useAmp,
    bool runBwd,
    int numIters,
    bool usePlugin) {
  input.setCalcGrad(false);
  network->eval();
  if (useAmp) {
    fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
  } else {
    fl::OptimMode::get().setOptimLevel(fl::OptimLevel::DEFAULT);
  }
  if (runBwd) {
    input.setCalcGrad(true);
    network->train();
  }
  auto benchmarkFunc = [&]() {
    network->zeroGrad();
    input.zeroGrad();
    fl::Variable output;
    if (usePlugin) {
      output = network->forward({input, fl::noGrad(duration)}).front();
    } else {
      output =
          fl::ext::forwardSequentialModuleWithPadMask(input, network, duration);
    }
    if (runBwd) {
      output.backward();
    }
  };
  // warmup
  for (int i = 0; i < 3; ++i) {
    benchmarkFunc();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < numIters; ++i) {
    benchmarkFunc();
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
             .count() /
      numIters;
}
} // namespace

int main(int argc, char** argv) {
  fl::init();
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  gflags::ParseCommandLineFlags(&argc, &argv, false);

  af::info();

  std::cout << "Loading architecture file from " << FLAGS_arch << std::endl;
  std::shared_ptr<fl::Module> network;
  bool usePlugin = fl::lib::endsWith(FLAGS_arch, ".so");
  if (usePlugin) {
    network = fl::ext::ModulePlugin(FLAGS_arch)
                  .arch(FLAGS_in_features, FLAGS_out_channels);
  } else {
    network = fl::ext::buildSequentialModule(
        FLAGS_arch, FLAGS_in_features, FLAGS_out_channels);
  }
  std::cout << network->prettyString() << std::endl;
  // Consider an audio input of 15 sec
  auto input = fl::Variable(
      af::randu(1500, FLAGS_in_features, 1, FLAGS_batchsize), true);
  auto duration = af::constant(1500, 1, FLAGS_batchsize).as(af::dtype::s64);

  std::cout << (FLAGS_use_amp ? "U" : "Not u") << "sing AMP" << std::endl;
  std::cout << "Fwd : "
            << run(network,
                   input,
                   duration,
                   FLAGS_use_amp /* useAmp */,
                   false /* runBwd */,
                   FLAGS_num_iters,
                   usePlugin)
            << " msec" << std::endl;
  std::cout << "Fwd+Bwd : "
            << run(network,
                   input,
                   duration,
                   FLAGS_use_amp /* useAmp */,
                   true /* runBwd */,
                   FLAGS_num_iters,
                   usePlugin)
            << " msec" << std::endl;

  return 0;
}
