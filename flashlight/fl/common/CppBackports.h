/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file common/CppBackports.h
 *
 * Backports of simple post-C++14 features.
 */

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#pragma once

namespace fl {
namespace cpp {

// ========== enum hashing ==========

// This is a temporary solution for a bug that is fixed in gcc 6.1+. After
// upgrade to that version+, this can be removed. Read more:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60970
struct EnumHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

// Hasher conditional on whether the type is an enum
template <typename Key>
using HashType = typename std::
    conditional<std::is_enum<Key>::value, EnumHash, std::hash<Key>>::type;

template <typename Key, typename T>
using fl_unordered_map = std::unordered_map<Key, T, HashType<Key>>;
template <typename T>
using fl_unordered_set = std::unordered_set<T, HashType<T>>;

} // namespace cpp
} // namespace fl
