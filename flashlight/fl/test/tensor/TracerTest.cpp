/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <iomanip>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/backend/trace/DefaultTracer.h"
#include "flashlight/fl/tensor/backend/trace/TensorTracer.h"

using namespace fl;

template <typename... Args>
std::string traceToString(DefaultTracer& tracer, Args&&... args) {
  tracer.trace(std::forward<Args>(args)...);
  std::stringstream ss;
  auto& stream = tracer.getStream();
  ss << stream.rdbuf();
  stream.clear();
  return ss.str();
}

TEST(TracerTest, DefaultTracerTypes) {
  DefaultTracer tracer(std::make_unique<std::stringstream>());
  std::stringstream ss;

  ASSERT_EQ(
      tracer.toTraceString(fl::rand({2, 2})),
      R"({"tensor": {"shape": [2, 2], "type": "f32"}})");
  ASSERT_EQ(
      tracer.toTraceString(fl::full({9}, 4, fl::dtype::u32)),
      R"({"tensor": {"shape": [9], "type": "u32"}})");

  Index i = fl::range(3, 6);
  Index j = 4;
  Index k = fl::arange({5});
  Index l = fl::range(0, fl::end);
  std::string _i = R"({"range": {"start": 3, "end": 5, "stride": 1}})";
  std::string _j = R"(4)";
  std::string _k = R"({"tensor": {"shape": [5], "type": "f32"}})";
  std::string _l = R"({"range": {"start": 0, "end": "end", "stride": 1}})";
  ASSERT_EQ(tracer.toTraceString(i), _i);
  ASSERT_EQ(tracer.toTraceString(j), _j);
  ASSERT_EQ(tracer.toTraceString(k), _k);
  ASSERT_EQ(tracer.toTraceString(l), _l);

  ss << "[" << _i << ", " << _j << ", " << _k << ", " << _l << "]";
  ASSERT_EQ(tracer.toTraceString({i, j, k, l}), ss.str());

  ASSERT_EQ(tracer.toTraceString(std::vector<int>({3, 4, 5})), "[3, 4, 5]");

  ASSERT_EQ(
      tracer.toTraceString(std::vector<std::pair<int, int>>({{1, 2}, {3, 4}})),
      "[[1, 2], [3, 4]]");

  ASSERT_EQ(
      tracer.toTraceString(
          std::vector<Tensor>({fl::rand({2, 3}), fl::full({5, 5}, 4)})),
      R"([{"tensor": {"shape": [2, 3], "type": "f32"}}, )"
      R"({"tensor": {"shape": [5, 5], "type": "s32"}}])");
  // Primitives
  ASSERT_EQ(tracer.toTraceString(6.2), "6.2");
  ASSERT_EQ(tracer.toTraceString(2), "2");
  ASSERT_EQ(tracer.toTraceString(false), R"("false")");
  ASSERT_EQ(tracer.toTraceString(31u), "31");
  ASSERT_EQ(tracer.toTraceString(123456l), "123456");
  ASSERT_EQ(tracer.toTraceString(123456789ul), "123456789");
  ASSERT_EQ(tracer.toTraceString(1234567890ull), "1234567890");

  ASSERT_EQ(tracer.toTraceString(fl::dtype::f32), R"("f32")");
  ASSERT_EQ(tracer.toTraceString(Dim(2)), "2");
  ASSERT_EQ(tracer.toTraceString(Shape({2, 3, 4})), "[2, 3, 4]");
  ASSERT_EQ(
      tracer.toTraceString(SortMode::Descending), R"("SortMode::Descending")");
  ASSERT_EQ(tracer.toTraceString(PadType::Constant), R"("PadType::Constant")");
  ASSERT_EQ(
      tracer.toTraceString(MatrixProperty::Transpose),
      R"("MatrixProperty::Transpose")");
  ASSERT_EQ(
      tracer.toTraceString(SortMode::Descending), R"("SortMode::Descending")");
  ASSERT_EQ(tracer.toTraceString(Location::Device), R"("Location::Device")");
  ASSERT_EQ(
      tracer.toTraceString(StorageType::Dense), R"("StorageType::Dense")");
}

TEST(TracerTest, ArgStructure) {
  DefaultTracer tracer(std::make_unique<std::stringstream>());

  ASSERT_EQ(
      traceToString(
          tracer,
          "testFunc",
          TracerBase::ArgumentList{},
          TracerBase::ArgumentList{},
          TracerBase::ArgumentList{}),
      R"({"testFunc": {"args": {}, "inputs": {}, "outputs": {}}})"
      "\n");
  ASSERT_EQ(
      traceToString(
          tracer,
          "testFunc",
          TracerBase::ArgumentList{{"arg1", fl::rand({3, 3})}, {"arg2", 2}},
          TracerBase::ArgumentList{},
          TracerBase::ArgumentList{{"arg3", Shape({1, 2, 3})}}),
      R"({"testFunc": {"args": {"arg1": {"tensor": {"shape": [3, 3], )"
      R"("type": "f32"}}, "arg2": 2}, "inputs": {}, )"
      R"("outputs": {"arg3": [1, 2, 3]}}})"
      "\n");
}

const std::string hexRegex("0[xX][0-9a-fA-F]+");

TEST(TracerTest, TensorTracer) {
  TensorTracer tracer(std::make_unique<std::stringstream>());

  auto tensor = fl::full({4, 8}, 7., fl::dtype::f64);

  std::string matchStr =
      R"#(\{"tensor": \{"shape": \[4, 8\], "type": "f64", "device": [0-9], )#"
      R"#("backend": "([^\s]+)", "memlocation": ")#" +
      hexRegex + R"("\}\})";
  std::string traceOutput = tracer.toTraceString(tensor);

  std::smatch m; // regex matches
  ASSERT_TRUE(std::regex_match(traceOutput, m, std::regex(matchStr)));
  ASSERT_EQ(m.size(), 2); // exactly one match
  ASSERT_EQ(m.str(), traceOutput); // should match the whole string

  // mem address should be the same as the tensor's address
  std::stringstream ss;
  ASSERT_TRUE(std::regex_search(traceOutput, m, std::regex(hexRegex)));
  ss << std::hex << m.str();
  unsigned long addr;
  ss >> addr;
  ss.clear();
  ASSERT_EQ(addr, reinterpret_cast<unsigned long>(tensor.device<void>()));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
