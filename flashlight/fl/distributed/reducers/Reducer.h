/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fl {

class Variable;

/**
 * An interface for creating tensor reduction algorithms/rules.
 *
 * In flashlight, a Reducer instance is typically used for gradient
 * synchronization across devices/processes during training, although the API is
 * general.
 */
class Reducer {
 public:
  virtual ~Reducer() = default;

  /**
   * Have the Reducer ingest a Variable. What happens next is
   * implementation-specific; the implementation may cache the value,
   * process/synchronize immediately, or ignore the value.
   *
   * @param[in] var a Variable to be ingested
   */
  virtual void add(Variable& var) = 0;

  /**
   * Forces a reduction/synchronization of the Reducer.
   * For some implementations, this may be a no-op if the Reducer immediately
   * processes or synchronizes all gradients that are added.
   */
  virtual void finalize() = 0;
};

} // namespace fl
