/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/distributed/reducers/Reducer.h"

#include <arrayfire.h>

namespace fl {

class Variable;

/**
 * A Reducer which coalesces added Variables in a cache until some maximum
 * cache size is reached, after which all Variables in the cache are reduced
 * and the cache is emptied.
 *
 * Since the Reducer executes ``allReduceMultiple`` operations asynchronously,
 * to guarantee that synchronized values are available after reduction,
 * ``finalize`` must be called before using a given value.
 */
class CoalescingReducer : public Reducer {
  /// A scale by which to scale reduced gradients
  double scale_;
  /// Whether or not the distributed synchronization operates in a separate
  /// compute stream asynchronously to the ArrayFire stream
  bool async_{true};
  /// Determines if the coalesced batch of gradients is put into
  /// contiguous memory before being synchronized
  bool contiguous_{true};
  /// The threshold at which the cache will be flushed and its contents
  /// synchronized, in bytes
  const std::size_t cacheThresholdBytes_;
  /// A cache that stores coalesced gradients.
  std::vector<Variable> cache_;
  /// The current cache size, in bytes
  std::size_t currCacheSize_{0};

 public:
  /**
   * Creates a new coalescing reducer.
   *
   * @param[in] cache threshold at which the cache will be flushed
   *  and its contents synchronized, in bytes
   * @param[in] async determines whether or not the distributed compute stream
   * runs asynchronously to the AF stream.
   * @param[in] contiguous forces synchronization of the set of Variables
   * to occur in a contiguous buffer, which may improve performance.
   */
  CoalescingReducer(double scale, bool async, bool contiguous);

  /**
   * Destroy the Reducer. Calls `finalize()` before returning.
   */
  ~CoalescingReducer() override;

  /**
   * Add a ``Variable`` to ``Reducer``. Behaves as follows:
   * - if the ``Variable`` exceeds the size of the coalescing cache, call
   *   ``allReduce`` immediately to synchronize.
   * - if the ``Variable`` is smaller than the cache and adding it would push
   *   the cache oversize, flush the cache and synchronize with
   *   ``allReduceMultiple``
   * - otherwise, add the ``Variable`` to the cache.
   */
  void add(Variable& var) override;

  /**
   * Flush any remaining ``Variable``s in the cache and synchronize.
   */
  void finalize() override;

 private:
  /**
   * Synchronize the existing set of Variables with ``allReduceMultiple`` and
   * reset the cache.
   */
  void flush();

  /**
   * Synchronize the distributed computation stream with the existing AF
   * computation stream in a way that doesn't block the main host thread.
   */
  void synchronize();
};

} // namespace fl
