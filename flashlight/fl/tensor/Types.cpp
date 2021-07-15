/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Types.h"

#include <unordered_map>

namespace fl {

const std::unordered_map<dtype, std::string> kTypeToString = {
    {dtype::f16, "f16"},
    {dtype::f32, "f32"},
    {dtype::f64, "f64"},
    {dtype::b8, "b8"},
    {dtype::s16, "s16"},
    {dtype::s32, "s32"},
    {dtype::s64, "s64"},
    {dtype::u8, "u8"},
    {dtype::u16, "u16"},
    {dtype::u32, "u32"},
    {dtype::u64, "u64"},
};

const std::string& dtypeToString(dtype type) {
  return kTypeToString.at(type);
}

std::ostream& operator<<(std::ostream& ostr, const dtype& s) {
  ostr << dtypeToString(s);
  return ostr;
}

} // namespace fl
