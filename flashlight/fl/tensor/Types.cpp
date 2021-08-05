/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Types.h"

#include <stdexcept>
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

size_t getTypeSize(dtype type) {
  switch (type) {
    case dtype::f16:
      return sizeof(float) / 2;
    case dtype::f32:
      return sizeof(float);
    case dtype::f64:
      return sizeof(double);
    case dtype::b8:
      return sizeof(unsigned char);
    case dtype::s16:
      return sizeof(short);
    case dtype::s64:
      return sizeof(long long);
    case dtype::s32:
      return sizeof(int);
    case dtype::u8:
      return sizeof(unsigned char);
    case dtype::u16:
      return sizeof(unsigned short);
    case dtype::u32:
      return sizeof(unsigned);
    case dtype::u64:
      return sizeof(unsigned long long);
    default:
      throw std::invalid_argument("getTypeSize - invalid type queried.");
  }
}

const std::string& dtypeToString(dtype type) {
  return kTypeToString.at(type);
}

std::ostream& operator<<(std::ostream& ostr, const dtype& s) {
  ostr << dtypeToString(s);
  return ostr;
}

} // namespace fl
