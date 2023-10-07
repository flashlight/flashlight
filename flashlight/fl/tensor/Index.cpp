/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Index.h"

namespace fl {

range::range(const Dim& i) : range(0, i) {}

range::range(const Dim& start, const idx& end)
    : range(start, end, /* stride */ kDefaultStride) {}

range::range(const Dim& start, const idx& end, const Dim stride)
    : start_(start),
      end_(
          std::holds_alternative<fl::end_t>(end)
              ? std::nullopt
              : std::optional<Dim>(std::get<Dim>(end))),
      stride_(stride) {}

Dim range::start() const {
  return start_;
}

const std::optional<Dim>& range::end() const {
  return end_;
}

Dim range::endVal() const {
  if (end_.has_value()) {
    return end_.value();
  }
  throw std::runtime_error("[range::endVal] end is end_t");
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
  : type_(range == span ? detail::IndexType::Span : detail::IndexType::Range),
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
