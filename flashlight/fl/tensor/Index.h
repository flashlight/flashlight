/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>
#include <variant>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Represents the imaginary index _after_ the last index along an axis of a
 * tensor. We have this special case because the `range::end` is exclusive.
 */
struct end_t {};

// A static instance of end_t for convenience, e.g., one can use it in
// `range(0, end)` to index all elements along certain axis.
static const end_t end = end_t();

/**
 * An entity representing a contiguous or strided sequence of indices.
 *
 * Assuming an axis has N elements, this is the mapping from negative to
 * positive indices:
 *  -------------------------
 *  | -N | -N+1 | ... |  -1 |
 *  -------------------------
 *  |  0 |    1 | ... | N-1 |
 *  -------------------------
 */
class FL_API range {
  using idx = std::variant<end_t, Dim>;
  static constexpr Dim kDefaultStride = 1;

  Dim start_{0};
  // end is exclusive; std::nullopt means including the last element
  std::optional<Dim> end_{std::nullopt};
  Dim stride_{kDefaultStride};

 public:
  /**
   * Default ctor.
   */
  range() = default;

  /**
   * Construct a range with the indices [0, idx) (i.e. [0, idx - 1])
   *
   * @param[in] idx the end index of the range, which will start from 0
   */
  explicit range(const Dim& idx);

  /**
   * Construct a range with the indices [start, end) (i.e. [start, end - 1])
   *
   * @param[in] start the starting index of the range
   * @param[in] end the end index of the range, which will start from 0
   */
  range(const Dim& start, const idx& end);

  /**
   * Construct a range with the indices [start, end) (i.e. [start, end - 1])
   * with the given stride.
   *
   * @param[in] start the starting index of the range
   * @param[in] end the end index of the range, which will start from 0
   * @param[in] stride the interval over which successive range elements appear
   */
  range(const Dim& start, const idx& end, const Dim stride);

  Dim start() const;
  // std::nullopt represents `end_t`
  const std::optional<Dim>& end() const;
  // throw if end is `end_t`
  Dim endVal() const;
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
struct FL_API Index {
  using IndexVariant = std::variant<Dim, range, Tensor>;

 private:
  // The type of indexing operator.
  detail::IndexType type_;

  // Underlying data referred to by the index
  IndexVariant index_;

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

  IndexVariant getVariant() const {
    return index_;
  }
};

} // namespace fl
