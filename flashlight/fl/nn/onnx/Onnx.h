/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace onnx {
struct Graph;
}

namespace fl {
namespace detail {

/**
 * Convert an ONNX graph into a Flashlight module.
 *
 * @param[in] graph the ONNX graph to convert
 * @return a Flashlight module that will execute the computation
 */
std::unique_ptr<Module> parseOnnxGraph(const onnx::Graph& graph);
} // namespace detail

/**
 * Load a serialized ONNX model and get a usable Flashlight module.
 *
 * @param[in] path the path to the serialized .onnx file.
 * @return a module corresponding to the loaded proto
 */
std::unique_ptr<Module> loadOnnxModuleFromPath(fs::path path);

} // namespace fl
