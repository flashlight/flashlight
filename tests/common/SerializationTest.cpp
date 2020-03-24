/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/common/Serialization.h"

// ========== utility functions ==========

template <typename T>
std::string saveToString(const T& t) {
  std::ostringstream oss(std::ios::binary);
  fl::save(oss, t);
  return oss.str();
}

template <typename T>
void loadFromString(const std::string& s, T& t) {
  std::istringstream iss(s, std::ios::binary);
  fl::load(iss, t);
}

template <typename T>
void checkRoundTrip(const T& t0) {
  T t1;
  loadFromString(saveToString(t0), t1);
  ASSERT_EQ(t0, t1);
}

// ========== basic serialization of structs ==========

struct Basic {
  int x;
  double y;
  std::string s;
  std::vector<int> v;

  FL_SAVE_LOAD(x, y, s, v)

  Basic() = default;

  Basic(int x, double y, std::string s, std::vector<int> v)
      : x(x), y(y), s(std::move(s)), v(std::move(v)) {}

  bool operator==(const Basic& o) const {
    return std::tie(x, y, s, v) == std::tie(o.x, o.y, o.s, o.v);
  }
};

TEST(SerializationTest, Basic) {
  checkRoundTrip(Basic{3, 5.5, "asdf", {2, 4, 6}});
}

// ========== versioning compatibility ===========

struct BasicV1 {
  int x;
  double y;
  float z{-1.0f};
  std::string s;
  std::vector<int> v;

  FL_SAVE_LOAD(x, y, fl::versioned(z, 1), s, v)

  BasicV1() = default;

  BasicV1(int x, double y, float z, std::string s, std::vector<int> v)
      : x(x), y(y), z(z), s(std::move(s)), v(std::move(v)) {}

  bool operator==(const BasicV1& o) const {
    return std::tie(x, y, z, s, v) == std::tie(o.x, o.y, o.z, o.s, o.v);
  }
};

CEREAL_CLASS_VERSION(BasicV1, 1);

TEST(SerializationTest, Versions) {
  checkRoundTrip(BasicV1{3, 5.5, 1.5f, "asdf", {2, 4, 6}});

  Basic v0{3, 5.5, "asdf", {2, 4, 6}};
  BasicV1 v1;
  loadFromString(saveToString(v0), v1);
  ASSERT_EQ(v0.x, v1.x);
  ASSERT_EQ(v0.y, v1.y);
  ASSERT_EQ(-1.0f, v1.z);
  ASSERT_EQ(v0.s, v1.s);
  ASSERT_EQ(v0.v, v1.v);
}

// sanity check for testing -- useless in practice
struct NestedVersioned {
  int x{1};
  int y{2};
  int z{3};
  int w{4};

  FL_SAVE_LOAD(
      fl::versioned(fl::versioned(x, 0, 2), 1, 3), /* 2 \in [1, 2] */
      fl::versioned(fl::versioned(y, 3), 0, 5), /* 2 \not\in [3, 5] */
      fl::versioned(fl::versioned(z, 2, 2), 1), /* 2 \in [2, 2] */
      fl::versioned(fl::versioned(w, 0), 0, 1)) /* 2 \not\in [0, 1] */
};

CEREAL_CLASS_VERSION(NestedVersioned, 2)

TEST(SerializationTest, NestedVersioned) {
  NestedVersioned t0;
  t0.x = 5;
  t0.y = 6;
  t0.z = 7;
  t0.w = 8;
  NestedVersioned t1;
  loadFromString(saveToString(t0), t1);
  ASSERT_EQ(t1.x, 5);
  ASSERT_EQ(t1.y, 2);
  ASSERT_EQ(t1.z, 7);
  ASSERT_EQ(t1.w, 4);
}

// ========== conversions using serializeAs ==========

struct SerializeIntAsFloat {
  int x;

  FL_SAVE_LOAD(fl::serializeAs<float>(x))

  bool operator==(const SerializeIntAsFloat& o) const {
    return x == o.x;
  }
};

struct SerializeFloatAsInt {
  float x;

  FL_SAVE_LOAD(fl::serializeAs<int>(x))
};

struct SerializeLongAsSqrt {
  long x;

  FL_SAVE_LOAD(fl::serializeAs<double>(
      x,
      [](const long& x) -> double { return std::sqrt(x); },
      [](double y) -> long { return std::lround(y * y); }))

  bool operator==(const SerializeLongAsSqrt& o) const {
    return x == o.x;
  }
};

TEST(SerializationTest, Conversions) {
  checkRoundTrip(SerializeIntAsFloat{12345});

  SerializeFloatAsInt fi{3.3f};
  loadFromString(saveToString(fi), fi);
  ASSERT_EQ(fi.x, 3.0f); // truncated due to static_cast

  checkRoundTrip(SerializeLongAsSqrt{13579});
}

// ========== passing temporary rvalues to FL_SAVE_LOAD ==========

// this will compile. saving will write x + 1 as expected, but
// loading will read to a temporary which is discarded.
struct SerializeNoOpTemporary {
  int x{0};

  FL_SAVE_LOAD(x + 1)
};

struct SerializeNoOpTemporaryInspect {
  int x;

  FL_SAVE_LOAD(x)
};

TEST(SerializationTest, TemporaryNoOp) {
  SerializeNoOpTemporary t0;
  t0.x = 3;
  auto s = saveToString(t0); // saves 4

  SerializeNoOpTemporaryInspect ins;
  loadFromString(s, ins); // loads 4
  ASSERT_EQ(ins.x, 4);

  SerializeNoOpTemporary t1;
  loadFromString(s, t1); // doesn't actually load
  ASSERT_EQ(t1.x, 0);
}

// multiplies by 10 before saving, adds 3 before loading
template <typename T>
struct WeirdTransform {
  T&& x;

  template <class Archive>
  void save(Archive& ar) const {
    ar(x * 10);
  }

  template <class Archive>
  void load(Archive& ar) {
    fl::cpp::decay_t<T> y;
    ar(y);
    x = y + 3;
  }
};

template <typename T>
WeirdTransform<T> weirdTransform(T&& t) {
  return WeirdTransform<T>{std::forward<T>(t)};
}

struct SerializeViaTemporary {
  int x;
  int y;
  int z;

  FL_SAVE_LOAD(weirdTransform(x), y, weirdTransform(z))
};

TEST(SerializationTest, TemporaryRvalues) {
  SerializeViaTemporary t{5, 6, 7};
  loadFromString(saveToString(t), t);
  ASSERT_EQ(t.x, 53);
  ASSERT_EQ(t.y, 6);
  ASSERT_EQ(t.z, 73);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
