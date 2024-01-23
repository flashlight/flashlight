/*
 * Copyright (c) Facebook, Inc. 6and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ostream>
#include <string>

#include "flashlight/fl/common/Defines.h"

namespace fl {

enum class dtype {
  f16 = 0, // 16-bit float
  f32 = 1, // 32-bit float
  f64 = 2, // 64-bit float
  b8 = 3, // 8-bit boolean
  s16 = 4, // 16-bit signed integer
  s32 = 5, // 32-bit signed integer
  s64 = 6, // 64-bit signed integer
  u8 = 7, // 8-bit unsigned integer
  u16 = 8, // 16-bit unsigned integer
  u32 = 9, // 32-bit unsigned integer
  u64 = 10 // 64-bit unsigned integer
  // TODO: add support for complex-valued tensors? (AF)
};

/**
 * Returns the size of the type in bytes.
 *
 * @param[in] type the input type to query.
 */
FL_API size_t getTypeSize(dtype type);

/**
 * Convert a dtype to its string representation.
 */
FL_API const std::string& dtypeToString(dtype type);

/**
 * Converts string to a Flashlight dtype
 *
 * @param[in] string type name as a string.
 *
 * @return returns the corresponding Flashlight dtype
 */
FL_API fl::dtype stringToDtype(const std::string& string);

/**
 * Write a type's string representation to an output stream.
 */
FL_API std::ostream& operator<<(std::ostream& ostr, const dtype& s);

template <typename T>
struct dtype_traits;

#define FL_TYPE_TRAIT(BASE_TYPE, DTYPE, CONSTANT_TYPE, STRING_NAME)    \
  template <>                                                          \
  struct FL_API dtype_traits<BASE_TYPE> {                                     \
    static const dtype fl_type = DTYPE; /* corresponding dtype */      \
    static const dtype ctype = CONSTANT_TYPE; /* constant init type */ \
    typedef BASE_TYPE base_type;                                       \
    static const char* getName() {                                     \
      return STRING_NAME;                                              \
    }                                                                  \
  }

FL_TYPE_TRAIT(float, dtype::f32, dtype::f32, "float");
FL_TYPE_TRAIT(double, dtype::f64, dtype::f32, "double");
FL_TYPE_TRAIT(int, dtype::s32, dtype::s32, "int");
FL_TYPE_TRAIT(unsigned, dtype::u32, dtype::u32, "unsigned int");
FL_TYPE_TRAIT(char, dtype::b8, dtype::s32, "char");
FL_TYPE_TRAIT(unsigned char, dtype::u8, dtype::u32, "unsigned char");
FL_TYPE_TRAIT(long, dtype::s64, dtype::s32, "long int");
FL_TYPE_TRAIT(unsigned long, dtype::u64, dtype::u32, "unsigned long");
FL_TYPE_TRAIT(long long, dtype::s64, dtype::s64, "long long");
FL_TYPE_TRAIT(unsigned long long, dtype::u64, dtype::u64, "unsigned long long");
FL_TYPE_TRAIT(bool, dtype::u8, dtype::u8, "bool");
FL_TYPE_TRAIT(short, dtype::s16, dtype::s16, "short");
FL_TYPE_TRAIT(unsigned short, dtype::u16, dtype::u16, "short");

} // namespace fl
