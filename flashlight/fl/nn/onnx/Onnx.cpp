/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/onnx/Onnx.h"

#include <iostream> // delete me
#include <memory>

#include <onnx/onnx_pb.h>
#include <onnx/proto_utils.h>

#include "flashlight/fl/nn/modules/Transform.h"

namespace fl {

std::shared_ptr<Module> loadOnnxModule() {
  // TODO: test compilation
  char* bytes = 0;
  int nbytes = 0;
  std::unique_ptr<onnx::ModelProto> model(new onnx::ModelProto());
  onnx::ParseProtoFromBytes(model.get(), bytes, nbytes);

  std::cout << model->graph().input().size() << std::endl;

  return std::make_shared<Transform>([](const Variable& var) { return var; });
}

} // namespace fl
