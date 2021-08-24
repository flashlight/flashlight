/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <variant>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Represents the last index along an axis of a tensor.
 */
struct end_t {
  operator Dim() const {
    return -1;
  }
};

// A static alias for end that can properly decay to an index type
static const end_t end = end_t();

/**
 * An entity representing a contiguous or strided sequence of indices.
 */
class range {
  using idx = std::variant<end_t, Dim>;

  Dim start_{0};
  Dim end_{fl::end};
  Dim stride_{1};

 public:
  /**
   * Default ctor.
   */
  range() = default;

  /**
   * Construct a range with the indices [0, idx) (i.e. [0, idx - 1])
   */
  explicit range(idx idx);

  /**
   * Construct a range with the indices [start, end) (i.e. [start, end - 1])
   */
  range(idx start, idx end);

  /**
   * Construct a range with the indices [start, end) (i.e. [start, end - 1])
   * with the given stride.
   */
  range(idx start, idx end, Dim stride);

  Dim start() const;
  Dim end() const;
  Dim stride() const;
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
  std::variant<Dim, range, Tensor> index_;

  // Intentionally private
  Index() = default;

 public:
  /* implicit */ Index(const Tensor& tensor);
  /* implicit */ Index(const range& range);
  /* implicit */ Index(const Dim idx);

  /**
   * Default copy assignment operator.
   */
  Index& operator=(const Index&) = default;

  /**
   * Move constructor - moves the index data.
   */
  Index(Index&& index) noexcept;
  Index(const Index& index) = default;

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
