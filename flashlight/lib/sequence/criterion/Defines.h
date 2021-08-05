/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fl {
namespace lib {
namespace seq {

enum class CriterionScaleMode {
  NONE = 0,
  INPUT_SZ = 1,
  INPUT_SZ_SQRT = 2,
  TARGET_SZ = 3,
  TARGET_SZ_SQRT = 4,
};
} // namespace seq
} // namespace lib
} // namespace fl
