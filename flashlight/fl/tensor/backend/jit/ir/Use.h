/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
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
  // TODO don't store user as a node reference.
  // Currently this is a reference instead of `NodePtr` because
  // 1. We cannot have circular reference with `std::shared_ptr`
  // 2. To pass in the user, we'll have to somehow convert a `Node* this` to a
  // `NodePtr` in the `Node` constructor code path.
  Node& user_;
  const unsigned inputIdx_;

 public:
  Use(Node& user, unsigned inputIdx);
  // no copy/move
  Use(const Use&) = delete;
  Use& operator=(const Use&) = delete;
  Use(Use&&) = delete;
  Use& operator=(Use&&) = delete;

  unsigned inputIdx() const;
  Node& user() const;
};

} // namespace fl
