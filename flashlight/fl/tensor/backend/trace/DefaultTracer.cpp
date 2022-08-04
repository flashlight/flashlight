/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/trace/DefaultTracer.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace fl {
namespace {

std::string stringToBracketedList(const Shape& shape) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.ndim(); ++i) {
    ss << shape[i] << (i < shape.ndim() - 1 ? ", " : "");
  }
  ss << "]";
  return ss.str();
}

} // namespace

DefaultTracer::DefaultTracer(std::unique_ptr<std::ostream> _stream)
    : TracerBase(std::move(_stream)) {}

std::string DefaultTracer::toTraceString(bool b) {
  return b ? R"("true")" : R"("false")";
}

std::string DefaultTracer::toTraceString(const Shape& shape) {
  std::stringstream ss;
  // ss << "{" << std::quoted("shape") << ": [";
  ss << "[";
  for (size_t i = 0; i < shape.ndim(); ++i) {
    ss << shape[i] << (i < shape.ndim() - 1 ? ", " : "");
  }
  ss << "]";
  return ss.str();
}

std::string DefaultTracer::toTraceString(const Tensor& tensor) {
  std::stringstream ss;
  ss << "{" << std::quoted("tensor") << ": {" << std::quoted("shape") << ": "
     << toTraceString(tensor.shape()) + ", " << std::quoted("type") << ": "
     << toTraceString(tensor.type()) << "}}";
  return ss.str();
}

std::string DefaultTracer::toTraceString(const dtype& type) {
  std::stringstream ss;
  ss << std::quoted(dtypeToString(type));
  return ss.str();
}

std::string DefaultTracer::toTraceString(const range& range) {
  std::stringstream ss;
  ss << "{" << std::quoted("range") << ": {" << std::quoted("start") << ": "
     << range.start() << ", " << std::quoted("end") << ": "
     << (range.end() == fl::end ? "\"end\"" : std::to_string(range.end()))
     << ", " << std::quoted("stride") << ": " << range.stride() << "}}";
  return ss.str();
}

std::string DefaultTracer::toTraceString(const Dim& dim) {
  std::stringstream ss;
  ss << dim;
  return ss.str();
}

std::string DefaultTracer::toTraceString(const Index& index) {
  std::stringstream ss;
  const Index::IndexVariant& _index = index.getVariant();
  std::visit([&ss, this](auto&& idx) { ss << toTraceString(idx); }, _index);
  return ss.str();
}

std::string DefaultTracer::toTraceString(const std::vector<Index>& indices) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < indices.size(); ++i) {
    ss << toTraceString(indices[i]) << (i < indices.size() - 1 ? ", " : "");
  }
  ss << "]";
  return ss.str();
}

std::string DefaultTracer::toTraceString(const std::vector<int>& vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << vec[i] << (i < vec.size() - 1 ? ", " : "");
  }
  ss << "]";
  return ss.str();
}

std::string DefaultTracer::toTraceString(
    const std::vector<std::pair<int, int>>& vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << "[" << vec[i].first << ", " << vec[i].second << "]"
       << (i < vec.size() - 1 ? ", " : "");
  }
  ss << "]";
  return ss.str();
}

std::string DefaultTracer::toTraceString(const std::vector<Tensor>& tensors) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < tensors.size(); ++i) {
    ss << toTraceString(tensors[i]) << (i < tensors.size() - 1 ? ", " : "");
  }
  ss << "]";
  return ss.str();
}

std::string DefaultTracer::toTraceString(const SortMode& sortMode) {
  // One day, use reflection
  static std::unordered_map<SortMode, std::string> map = {
      {SortMode::Descending, R"("SortMode::Descending")"},
      {SortMode::Ascending, R"("SortMode::Ascending")"}};
  return map[sortMode];
}

std::string DefaultTracer::toTraceString(const PadType& padType) {
  // One day, use reflection
  static std::unordered_map<PadType, std::string> map = {
      {PadType::Constant, R"("PadType::Constant")"},
      {PadType::Edge, R"("PadType::Edge")"},
      {PadType::Symmetric, R"("PadType::Symmetric")"}};
  return map[padType];
}

std::string DefaultTracer::toTraceString(const MatrixProperty& matrixProperty) {
  // One day, use reflection
  static std::unordered_map<MatrixProperty, std::string> map = {
      {MatrixProperty::None, R"("MatrixProperty::None")"},
      {MatrixProperty::Transpose, R"("MatrixProperty::Transpose")"}};
  return map[matrixProperty];
}

std::string DefaultTracer::toTraceString(const Location& location) {
  // One day, use reflection
  static std::unordered_map<Location, std::string> map = {
      {Location::Host, R"("Location::Host")"},
      {Location::Device, R"("Location::Device")"}};
  return map[location];
}

std::string DefaultTracer::toTraceString(const StorageType& storageType) {
  // One day, use reflection
  static std::unordered_map<StorageType, std::string> map = {
      {StorageType::Dense, R"("StorageType::Dense")"},
      {StorageType::CSR, R"("StorageType::CSR")"},
      {StorageType::CSC, R"("StorageType::CSC")"},
      {StorageType::COO, R"("StorageType::COO")"}};
  return map[storageType];
}

std::string DefaultTracer::traceArgumentList(ArgumentList args) {
  std::stringstream ss;
  ss << "{";
  for (auto iter = args.begin(); iter != args.end(); ++iter) {
    ss << std::quoted(iter->first) << ": ";
    std::visit(
        [&ss, this](auto&& arg) -> void { ss << toTraceString(arg); },
        iter->second);

    if (std::next(iter) != args.end()) {
      ss << ", ";
    }
  }
  ss << "}";
  return ss.str();
}

void DefaultTracer::trace(
    const std::string& funcName,
    ArgumentList args,
    ArgumentList inputs,
    ArgumentList outputs) {
  getStream() << "{" << std::quoted(funcName) << ": {" << std::quoted("args")
              << ": " << traceArgumentList(args) << ", "
              << std::quoted("inputs") << ": " << traceArgumentList(inputs)
              << ", " << std::quoted("outputs") << ": "
              << traceArgumentList(outputs) << "}"
              << "}" << std::endl;
  //  << ",";
}

} // namespace fl
