/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/**
 * @file common/Serialization-inl.h
 *
 * Implementation details, only to be included from Serialization.h
 */

#pragma once

namespace fl {
namespace detail {
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

// 0 arguments (no-op).
template <typename Archive>
void applyArchive(Archive& ar, const uint32_t version) {}

// 1 argument, not version-restricted.
template <typename Archive, typename Arg>
void applyArchive(Archive& ar, const uint32_t version, Arg&& arg) {
  ar(arg);
}

// 1 argument, version-restricted.
template <typename Archive, typename Arg>
void applyArchive(Archive& ar, const uint32_t version, Versioned<Arg> varg) {
  if (version >= varg.minVersion && version <= varg.maxVersion) {
    ar(std::forward<Arg>(varg.ref));
  }
}

// 2+ arguments (recurse).
template <typename Archive, typename Arg, typename... Args>
void applyArchive(
    Archive& ar,
    const uint32_t version,
    Arg&& arg,
    Args&&... args) {
  applyArchive(ar, version, arg);
  applyArchive(ar, version, std::forward<Args>(args)...);
}
} // namespace detail

template <typename T>
detail::Versioned<T>
versioned(T&& t, uint32_t minVersion, uint32_t maxVersion) {
  return detail::Versioned<T>{std::forward<T>(t), minVersion, maxVersion};
}

template <typename... Args>
void save(const std::string& filepath, const Args&... args) {
  std::ofstream ofs(filepath, std::ios::binary);
  save(ofs, args...);
}

template <typename... Args>
void save(std::ostream& ostr, const Args&... args) {
  cereal::BinaryOutputArchive ar(ostr);
  ar(args...);
}

template <typename... Args>
void load(const std::string& filepath, Args&... args) {
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
 * This workaround lets us use explicit versioning for af::array; if we'd used
 * `save(Archive& ar, const af::array& arr, const uint32_t version)` directly,
 * cereal would complain there are 2 ways to serialize integer types,
 * because af::array has an implicit ctor from a single `long long`.
 *
 * The trick we use here is that C++'s implicit conversion sequence permits
 * at most one user-defined conversion. Therefore `af::array` may be implicitly
 * converted to `AfArraySerializeProxy`, but `int` may not.
 *
 * For more info, see https://github.com/USCiLab/cereal/issues/132
 * and https://en.cppreference.com/w/cpp/language/implicit_conversion
 */
struct AfArraySerializeProxy {
  /* implicit */ AfArraySerializeProxy(const af::array& a) : val(a) {}
  /* implicit */ AfArraySerializeProxy(af::array&& a) : val(std::move(a)) {}
  af::array val;
};
} // namespace detail
} // namespace fl

namespace cereal {

// no versioning; simple and unlikely to ever change
template <class Archive>
void serialize(Archive& ar, af::dim4& dims) {
  ar(dims[0], dims[1], dims[2], dims[3]);
}

template <class Archive>
void save(
    Archive& ar,
    const fl::detail::AfArraySerializeProxy& proxy,
    const uint32_t /* version */) {
  const auto& arr = proxy.val;
  if (arr.issparse()) {
    throw cereal::Exception(
        "Serialzation of sparse af::array is not supported yet!");
  }
  std::vector<uint8_t> vec(arr.bytes());
  arr.host(vec.data());
  ar(arr.dims(), arr.type(), vec);
}

template <class Archive>
void load(Archive& ar, af::array& arr, const uint32_t /* version */) {
  af::dim4 dims;
  af::dtype ty;
  std::vector<uint8_t> vec;
  ar(dims, ty, vec);
  arr = af::array(dims, ty);
  arr.write(vec.data(), vec.size());
}

} // namespace cereal
