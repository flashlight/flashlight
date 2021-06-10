/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <variant>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Represents the last index along an axis of a tensor.
 */
constexpr int end = -1;

/**
 * An entity representing a contiguous or strided sequence of indices.
 */
class range {
  int start_{0};
  int end_{fl::end};
  int stride_{1};

 public:
  explicit range(int idx);
  range(int start, int end);
  range(int start, int end, int stride);
  int start() const;
  int end() const;
  int stride() const;
  bool operator==(const range& other) const;
  bool operator!=(const range& other) const;
};

// span is an instance of range
static const range span = range(-1, -1, 0);

namespace detail {

/**
 * Allowed indexing operators.
 */
enum class IndexType : int { Tensor = 0, Range = 1, Literal = 2, Span = 3 };

} // namespace detail

/**
 * An entity used to index a tensor.
 *
 * An index can be of a few different types, which are implicitly converted via
 * Index's constructors:
 * - fl::Tensor if doing advanced indexing, where elements from a tensor are
 *   indexed based on values in the indexing tensor
 * - fl::range which refers to a contiguous (or strided) sequence of indices.
 * - An index literal, which refers to a single subtensor of the tensor being
 *   indexed.
 */
class Index {
  // The type of indexing operator.
  detail::IndexType type_;
  // Underlying data referred to by the index
  std::variant<int, range, Tensor> index_;

  // Intentionally private
  Index() = default;

 public:
  /* implicit */ Index(const Tensor& tensor);
  /* implicit */ Index(const range& range);
  /* implicit */ Index(const int idx);

  /**
   * Get the index type for this index.
   *
   * @return the index type.
   */
  detail::IndexType type() const;

  /**
   * Returns true if the index represents a span.
   */
  bool isSpan() const;

  /**
   * Get the internal data for a particular Index. Parameterized by type. Will
   * throw as per std::variant if the type doesn't match this Index's underlying
   * type.
   */
  template <typename T>
  const T& get() const {
    return std::get<T>(index_);
  }

  template <typename T>
  T& get() {
    return std::get<T>(index_);
  }
};

} // namespace fl
