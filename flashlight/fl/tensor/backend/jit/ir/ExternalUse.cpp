/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.  */

#include "flashlight/fl/tensor/backend/jit/ir/ExternalUse.h"

namespace fl {

void ExternalUse::unlinkWithCurrentUsee() {
  usee_->externalUses_.erase(useIter_);
}

void ExternalUse::linkWithUsee(NodePtr usee) {
  usee_ = usee;
  useIter_ = usee->externalUses_.insert(usee->externalUses_.end(), this);
}

ExternalUse::ExternalUse(NodePtr usee) {
  linkWithUsee(usee);
}

ExternalUse::~ExternalUse() {
  unlinkWithCurrentUsee();
}

NodePtr ExternalUse::usee() const {
  return usee_;
}

void ExternalUse::setUsee(NodePtr newUsee) {
  unlinkWithCurrentUsee();
  linkWithUsee(newUsee);
}

} // namespace fl
