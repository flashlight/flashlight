/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/printer/ScopedPostEvalGraphvizPrinter.h"

#include <sstream>
#include <stdexcept>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/JitBackend.h"

namespace fl {

ScopedPostEvalGraphvizPrinter::ScopedPostEvalGraphvizPrinter(
    std::string filename,
    bool profile)
    : managedOfs_(std::make_unique<std::ofstream>(filename)),
      printer_(*managedOfs_) {
  hookToEvaluator(profile);
}

ScopedPostEvalGraphvizPrinter::ScopedPostEvalGraphvizPrinter(
    std::ostream& os,
    bool profile)
    : printer_(os) {
  hookToEvaluator(profile);
}

void ScopedPostEvalGraphvizPrinter::hookToEvaluator(bool profile) {
  auto& backend = defaultTensorBackend();
  if (backend.backendType() != TensorBackendType::Jit) {
    std::ostringstream oss;
    oss << "[ScopedGraphvizPrinter::ScopedGraphvizPrinter] "
        << "Expected default backend to be JitBackend, but got "
        << backend.backendType();
    throw std::runtime_error(oss.str());
  }
  auto& jitBackend = static_cast<JitBackend&>(backend);
  monitoredEvaluator_ = &jitBackend.evaluator();
  oldProfileState_ = monitoredEvaluator_->getProfilerState();
  monitoredEvaluator_->setProfilerState(profile);
  callbackHandle_ = monitoredEvaluator_->addPostEvalCallback(
      [this](NodePtr node, std::unordered_map<NodePtr, float> nodeToTotTimeMs) {
        printer_.printSubgraph(
            node, "scoped_printer_post_eval", nodeToTotTimeMs);
      });
}

ScopedPostEvalGraphvizPrinter::~ScopedPostEvalGraphvizPrinter() {
  monitoredEvaluator_->removePostEvalCallback(callbackHandle_);
  monitoredEvaluator_->setProfilerState(oldProfileState_);
}

} // namespace fl
