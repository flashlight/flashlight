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

#include <onnx/common/ir.h>
#include <onnx/common/ir_pb_converter.h>
#include <onnx/onnx_pb.h>
#include <onnx/proto_utils.h>

// Delete
#include <onnx/defs/printer.h>

#include "flashlight/fl/nn/modules/Transform.h"
#include "flashlight/fl/nn/modules/modules.h"

namespace fl {
namespace detail {

std::unique_ptr<Module> parseOnnxGraph(const onnx::Graph& graph) {
  auto seq = std::make_unique<Sequential>();

  auto iter = graph.begin();

  iter++; // the first operator is always the `input` op
  while (iter != graph.end()) {
    auto kind = iter->kind();
    std::cout << iter->kind().toString() << std::endl;
    std::cout << iter->name() << std::endl;
    if (kind == onnx::Symbol("Conv")) {
      // seq->add(fl::Conv2D());
    } else if (kind == onnx::Symbol("Relu")) {
      seq->add(ReLU());
    } else if (kind == onnx::Symbol("MaxPool")) {
      // seq->add(Pool2D());
    } else if (kind == onnx::Symbol("Gemm")) {
      // seq->add(Linear());
    } else if (kind == onnx::Symbol("Dropout")) {
      const auto inputs = iter->inputs();
      // std::cout << "Dropout -- inputs of size " << inputs.size() << " and
      // type "
      //           << inputs[0]->name() << std::endl;
      // seq->add(Dropout());
    } else if (kind == onnx::Symbol("Softmax")) {
      // seq->add(Dropout());
    } else {
      // Unimplemented: LRN

      // throw std::runtime_error(
      //     "parseOnnxGraph - no support for operator " +
      //     std::string(kind.toString()));
    }
    iter++;
  }
  return std::make_unique<Transform>([](const Variable& var) { return var; });
}

} // namespace detail

std::unique_ptr<Module> loadOnnxModuleFromPath(fs::path path) {
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

  // std::cout << model->graph().input().size() << std::endl;
  // std::cout << model->graph().node() << std::endl;

  std::unique_ptr<onnx::Graph> graph = onnx::ImportModelProto(*model);

  return detail::parseOnnxGraph(*graph);
}

} // namespace fl
