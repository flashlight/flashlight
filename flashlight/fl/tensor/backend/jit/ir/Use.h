/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fl {

class Node;

/**
 * A Use represents the use of some Node by one of its user nodes.
 * The used Node is always `use.user().inputs().at(use.inputIdx())`
 */
class Use {
  Node* const user_;
  const unsigned inputIdx_;

  // intentionally kept private
  Use(Node* user, unsigned inputIdx);

 public:
  static Use* create(Node* user, unsigned inputIdx);

  unsigned inputIdx() const;
  Node* user() const;
};

} // namespace fl
