/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/**
 * @file common/Serialization-inl.h
 *
 * Implementation details, only to be included from Serialization.h
 */

#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/tensor/TensorBase.h"

#pragma once

namespace fl {
namespace detail {

template <typename T>
using IsOutputArchive = std::is_base_of<cereal::detail::OutputArchiveBase, T>;

template <typename T>
using IsInputArchive = std::is_base_of<cereal::detail::InputArchiveBase, T>;

/**
 * Wrapper indicating that an expression should be serialized only if the
 * version is in a certain range.
 */
template <typename T>
struct Versioned {
  T&& ref;
  uint32_t minVersion;
  uint32_t maxVersion;
};

template <typename S, typename T>
struct SerializeAs {
  using T0 = std::decay_t<T>;
  T&& ref;
  std::function<S(const T0&)> saveConverter;
  std::function<T0(S)> loadConverter;
};

// 0 arguments (no-op).
template <typename Archive>
void applyArchive(Archive& ar, const uint32_t version) {}

// 1 argument, general case.
template <typename Archive, typename Arg>
void applyArchive(Archive& ar, const uint32_t version, Arg&& arg) {
  ar(std::forward<Arg>(arg));
}

// 1 argument, version-restricted.
template <typename Archive, typename T>
void applyArchive(Archive& ar, const uint32_t version, Versioned<T> varg) {
  if (version >= varg.minVersion && version <= varg.maxVersion) {
    applyArchive(ar, version, std::forward<T>(varg.ref));
  }
}

// 1 argument, with conversion, saving.
template <
    typename Archive,
    typename S,
    typename T,
    std::enable_if_t<IsOutputArchive<Archive>::value, int> = 0>
void applyArchive(Archive& ar, const uint32_t version, SerializeAs<S, T> arg) {
  if (arg.saveConverter) {
    applyArchive(ar, version, arg.saveConverter(arg.ref));
  } else {
    applyArchive(ar, version, static_cast<const S&>(arg.ref));
  }
}

// 1 argument, with conversion, loading.
template <
    typename Archive,
    typename S,
    typename T,
    std::enable_if_t<IsInputArchive<Archive>::value, int> = 0>
void applyArchive(Archive& ar, const uint32_t version, SerializeAs<S, T> arg) {
  using T0 = std::remove_reference_t<T>;
  S s;
  applyArchive(ar, version, s);
  if (arg.loadConverter) {
    arg.ref = arg.loadConverter(std::move(s));
  } else {
    arg.ref = static_cast<T0>(std::move(s));
  }
}

// 2+ arguments (recurse).
template <typename Archive, typename Arg, typename... Args>
void applyArchive(
    Archive& ar,
    const uint32_t version,
    Arg&& arg,
    Args&&... args) {
  applyArchive(ar, version, std::forward<Arg>(arg));
  applyArchive(ar, version, std::forward<Args>(args)...);
}

} // namespace detail

template <typename T>
detail::Versioned<T>
versioned(T&& t, uint32_t minVersion, uint32_t maxVersion) {
  return detail::Versioned<T>{std::forward<T>(t), minVersion, maxVersion};
}

template <typename S, typename T>
detail::SerializeAs<S, T> serializeAs(T&& t) {
  return detail::SerializeAs<S, T>{std::forward<T>(t), nullptr, nullptr};
}

template <typename S, typename T, typename SaveConvFn, typename LoadConvFn>
detail::SerializeAs<S, T>
serializeAs(T&& t, SaveConvFn saveConverter, LoadConvFn loadConverter) {
  return detail::SerializeAs<S, T>{
      std::forward<T>(t), std::move(saveConverter), std::move(loadConverter)};
}

template <typename... Args>
void save(const fs::path& filepath, const Args&... args) {
  std::ofstream ofs(filepath, std::ios::binary);
  save(ofs, args...);
}

template <typename... Args>
void save(std::ostream& ostr, const Args&... args) {
  cereal::BinaryOutputArchive ar(ostr);
  ar(args...);
}

template <typename... Args>
void load(const fs::path& filepath, Args&... args) {
  std::ifstream ifs(filepath, std::ios::binary);
  load(ifs, args...);
}

template <typename... Args>
void load(std::istream& istr, Args&... args) {
  cereal::BinaryInputArchive ar(istr);
  ar(args...);
}

namespace detail {
/**
 * This workaround lets us use explicit versioning for Tensor; if we'd used
 * `save(Archive& ar, const Tensor& tensor, const uint32_t version)` directly,
 * cereal would complain there are 2 ways to serialize integer types,
 * because Tensor has an implicit ctor from a single `long long`.
 *
 * The trick we use here is that C++'s implicit conversion sequence permits
 * at most one user-defined conversion. Therefore `Tensor` may be implicitly
 * converted to `FlTensorSerializeProxy`, but `int` may not.
 *
 * For more info, see https://github.com/USCiLab/cereal/issues/132
 * and https://en.cppreference.com/w/cpp/language/implicit_conversion
 */
template <typename T>
struct CerealSave {
  /* implicit */ CerealSave(const T& x) : val(x) {}
  const T& val;
};
} // namespace detail
} // namespace fl

namespace cereal {

// no versioning; simple and unlikely to ever change
template <class Archive>
void save(
    Archive& ar,
    const fl::detail::CerealSave<fl::Shape>& dims_,
    const uint32_t /* version */) {
  // TODO{fl::Tensor} -- check version, then op as dim4 (if version ==)
  const auto& dims = dims_.val;
  const std::vector<fl::Dim>& vec = dims.get();
  ar(vec);
}

template <class Archive>
void load(Archive& ar, fl::Shape& dims, const uint32_t /* version */) {
  // TODO{fl::Tensor} -- check version, then read dim4 into Shape (if version
  // ==)
  std::vector<fl::Dim> vec;
  ar(vec);
  dims = fl::Shape(vec);
}

template <class Archive>
void save(
    Archive& ar,
    const fl::detail::CerealSave<fl::Tensor>& tensor_,
    const uint32_t /* version */) {
  const auto& tensor = tensor_.val;
  // TODO{fl::Tensor}{sparse} figure out what to do here...
  if (tensor.isSparse()) {
    throw cereal::Exception(
        "Serialzation of sparse Tensor is not supported yet!");
  }
  std::vector<uint8_t> vec(tensor.bytes());
  tensor.host(vec.data());
  ar(tensor.shape(), tensor.type(), vec);
}

template <class Archive>
void load(Archive& ar, fl::Tensor& tensor, const uint32_t /* version */) {
  fl::Shape dims;
  fl::dtype ty;
  std::vector<uint8_t> vec;
  ar(dims, ty, vec);
  tensor = fl::Tensor::fromVector(dims, vec, ty);
}

} // namespace cereal
