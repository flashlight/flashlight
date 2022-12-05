/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <memory>

#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"
#include "flashlight/fl/tensor/backend/jit/printer/GraphvizPrinter.h"

namespace fl {

/**
 * A class that uses RAII to print out evaluated graph in some scope.
 */
class ScopedPostEvalGraphvizPrinter {
  // created only if user provided a filename
  std::unique_ptr<std::ofstream> managedOfs_;
  GraphvizPrinter printer_;
  Evaluator::PostEvalCallbackHandle callbackHandle_;
  bool oldProfileState_;
  Evaluator* monitoredEvaluator_;

  void hookToEvaluator(bool profile);

 public:
  // TODO doc
  ScopedPostEvalGraphvizPrinter(std::string filename, bool profile = false);
  ScopedPostEvalGraphvizPrinter(std::ostream& os, bool profile = false);
  ~ScopedPostEvalGraphvizPrinter();
};

} // namespace fl
