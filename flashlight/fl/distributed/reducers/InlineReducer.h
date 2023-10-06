/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/distributed/reducers/Reducer.h"

namespace fl {

class Variable;

/**
 * A Reducer which calls allReduce directly on gradients to process. All
 * synchronized gradients are scaled by a pre-specified factor.
 */
class FL_API InlineReducer : public Reducer {
  /// A scale by which to scale reduced gradients
  double scale_;

 public:
  /**
   * Creates a new InlineReducer with a given scaling factor
   *
   * @param[in] scale the factor by which to scale gradients after
   * synchronization
   */
  explicit InlineReducer(double scale);

  /**
   * Ingest a Variable and immediately call allReduce on it.
   *
   * @param[in] var the Variable to process for synchronization
   */
  void add(Variable& var) override;

  // no-op; no state
  void finalize() override {}
};

} // namespace fl
