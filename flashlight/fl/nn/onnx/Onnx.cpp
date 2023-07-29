/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/onnx/Onnx.h"

#include <fstream>
#include <iostream> // delete me
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

#include <onnx/onnx_pb.h>
#include <onnx/proto_utils.h>

#include "flashlight/fl/nn/modules/Transform.h"

namespace fl {

std::shared_ptr<Module> loadOnnxModuleFromPath(fs::path path) {
  std::fstream protoStream(path, std::ios::in | std::ios::binary);
  if (!protoStream) {
    throw std::invalid_argument(
        "loadOnnxModuleFromPath - file not found: " + path.string());
  }

  std::string data{
      std::istreambuf_iterator<char>{protoStream},
      std::istreambuf_iterator<char>{}};
  auto model = std::make_unique<onnx::ModelProto>();
  if (!onnx::ParseProtoFromBytes(model.get(), data.c_str(), data.size())) {
    throw std::runtime_error(
        "loadOnnxModuleFromPath - cannot parse ONNX proto: " + path.string());
  }

  std::cout << model->graph().input().size() << std::endl;

  return std::make_shared<Transform>([](const Variable& var) { return var; });
}

} // namespace fl
