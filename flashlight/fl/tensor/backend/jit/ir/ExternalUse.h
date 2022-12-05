/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.  */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * A Use represents the use of some Node by some non-Node entity (i.e., it's
 * "external" to the IR graph).
 */
class ExternalUse {
  NodePtr usee_;
  ExternalUseList::iterator useIter_; // enables fast removal

  void unlinkWithCurrentUsee();
  void linkWithUsee(NodePtr usee);

 public:
  ExternalUse(NodePtr usee);
  ~ExternalUse();
  // no copy
  ExternalUse(const ExternalUse&) = delete;
  ExternalUse& operator=(const ExternalUse&) = delete;
  // move only
  ExternalUse(ExternalUse&&) = default;
  ExternalUse& operator=(ExternalUse&&) = default;

  NodePtr usee() const;
  void setUsee(NodePtr newUsee);

  friend class Node; // enables faster direct usee replacement
};

} // namespace fl
