/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file common/CppBackports.h
 *
 * Backports of simple post-C++11 features.
 */

#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#pragma once

namespace fl {
namespace cpp {

// ========== type_traits convenience ==========

template <class T>
using remove_const_t = typename std::remove_const<T>::type;
template <class T>
using remove_volatile_t = typename std::remove_volatile<T>::type;
template <class T>
using remove_cv_t = typename std::remove_cv<T>::type;
template <class T>
using add_const_t = typename std::add_const<T>::type;
template <class T>
using add_volatile_t = typename std::add_volatile<T>::type;
template <class T>
using add_cv_t = typename std::add_cv<T>::type;

template <class T>
using remove_reference_t = typename std::remove_reference<T>::type;
template <class T>
using add_lvalue_reference_t = typename std::add_lvalue_reference<T>::type;
template <class T>
using add_rvalue_reference_t = typename std::add_rvalue_reference<T>::type;

template <class T>
using make_signed_t = typename std::make_signed<T>::type;
template <class T>
using make_unsigned_t = typename std::make_unsigned<T>::type;

template <class T>
using remove_extent_t = typename std::remove_extent<T>::type;
template <class T>
using remove_all_extents_t = typename std::remove_all_extents<T>::type;

template <class T>
using remove_pointer_t = typename std::remove_pointer<T>::type;
template <class T>
using add_pointer_t = typename std::add_pointer<T>::type;

template <class T>
using decay_t = typename std::decay<T>::type;
template <bool b, class T = void>
using enable_if_t = typename std::enable_if<b, T>::type;
template <bool b, class T, class F>
using conditional_t = typename std::conditional<b, T, F>::type;
template <class... T>
using common_type_t = typename std::common_type<T...>::type;
template <class T>
using underlying_type_t = typename std::underlying_type<T>::type;
template <class T>
using result_of_t = typename std::result_of<T>::type;

// ========== make_unique ==========

template <class T>
struct _unique_if {
  using _single_object = std::unique_ptr<T>;
};

template <class T>
struct _unique_if<T[]> {
  using _unknown_bound = std::unique_ptr<T[]>;
};

template <class T, std::size_t N>
struct _unique_if<T[N]> {
  using _known_bound = void;
};

template <class T, class... Args>
typename _unique_if<T>::_single_object make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename _unique_if<T>::_unknown_bound make_unique(std::size_t n) {
  using U = remove_extent_t<T>;
  return std::unique_ptr<T>(new U[n]());
}

template <class T, class... Args>
typename _unique_if<T>::_known_bound make_unique(Args&&...) = delete;

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

template <typename T>
using enum_unordered_set = std::unordered_set<T, EnumHash>;

template <typename KeyType, typename ValueType>
using enum_unordered_map = std::unordered_map<KeyType, ValueType, EnumHash>;

} // namespace cpp
} // namespace fl
