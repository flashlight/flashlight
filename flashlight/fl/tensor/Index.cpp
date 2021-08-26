/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Index.h"

namespace fl {

range::range(idx idx) : range(0, idx) {}

range::range(idx start, idx end) : range(start, end, /* stride */ 1) {}

range::range(idx start, idx end, Dim stride)
    : // fl::end decays to int
      start_(std::visit([](Dim idx) -> Dim { return idx; }, start)),
      // fl::end --> -1, else idx as Dim
      end_(
          std::holds_alternative<fl::end_t>(end)
              ? std::get<fl::end_t>(end)
              // If start == end, set start_ == end_, else end_ = end - 1
              : (std::get<Dim>(end) == start_ ? start_
                                              : std::get<Dim>(end) - 1)),
      stride_(stride) {}

Dim range::start() const {
  return start_;
}

Dim range::end() const {
  return end_;
}

Dim range::stride() const {
  return stride_;
}

bool range::operator==(const range& other) const {
  return start_ == other.start() && end_ == other.end() &&
      stride_ == other.stride();
}

bool range::operator!=(const range& other) const {
  return !this->operator==(other);
}

Index::Index(const Tensor& tensor)
    : type_(detail::IndexType::Tensor), index_(tensor) {}

Index::Index(const range& range)
    : type_(
          range == fl::range(-1, -1, 0) ? detail::IndexType::Span
                                        : detail::IndexType::Range),
      index_(range) {}

Index::Index(const Dim idx) : type_(detail::IndexType::Literal), index_(idx) {}

Index::Index(Index&& other) noexcept
    : type_(other.type_), index_(std::move(other.index_)) {}

detail::IndexType Index::type() const {
  return type_;
}

bool Index::isSpan() const {
  return type_ == detail::IndexType::Span;
}

} // namespace fl
