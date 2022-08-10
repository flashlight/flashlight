/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/printer/GraphvizPrinter.h"

#include <string>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

namespace fl {

namespace detail {

NameGenerator::NameGenerator() : namemCounter_(0) {}

std::string NameGenerator::genName(std::string prefix) {
  return prefix + std::to_string(namemCounter_++);
}

NodeNamer::NodeNamer(const std::string& prefix) : prefix_(prefix) {}

const std::string& NodeNamer::getName(NodePtr node) {
  if (nodeToName_.find(node) == nodeToName_.end()) {
    nodeToName_.emplace(node, nodeNameGenerator_.genName(prefix_));
  }
  return nodeToName_.at(node);
}

bool NodeNamer::contains(NodePtr node) const {
  return nodeToName_.find(node) != nodeToName_.end();
}

} // namespace detail

namespace {

const char* binopToStr(const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return "Add";
    case BinaryOp::Sub:
      return "Sub";
    case BinaryOp::Mul:
      return "Mul";
    case BinaryOp::Div:
      return "Div";
    case BinaryOp::Eq:
      return "Eq";
    case BinaryOp::Neq:
      return "Neq";
    case BinaryOp::Gt:
      return "Gt";
    case BinaryOp::Gte:
      return "Gte";
    case BinaryOp::Lt:
      return "Lt";
    case BinaryOp::Lte:
      return "Lte";
    case BinaryOp::Min:
      return "Min";
    case BinaryOp::Max:
      return "Max";
    case BinaryOp::Pow:
      return "Pow";
    case BinaryOp::Mod:
      return "Mod";
    case BinaryOp::And:
      return "And";
    case BinaryOp::Or:
      return "Or";
    case BinaryOp::Shl:
      return "Shl";
    case BinaryOp::Shr:
      return "Shr";
    case BinaryOp::BitAnd:
      return "BitAnd";
    case BinaryOp::BitOr:
      return "BitOr";
    case BinaryOp::BitXor:
      return "BitXor";
  }
  throw std::runtime_error("Unsupported binary operation type");
}

} // namespace

std::ostream& GraphvizPrinter::os() {
  return os_;
}

GraphvizPrinter& GraphvizPrinter::setEdgeColor(Color newColor) {
  this->edgeColor_ = newColor;
  return *this;
}

GraphvizPrinter& GraphvizPrinter::setNodeColor(Color newColor) {
  this->nodeColor_ = newColor;
  return *this;
}

void GraphvizPrinter::printBinaryNode(const BinaryNode& node) {
  os() << "label=\""
       << "BinaryNode"
       << "\\n"
       << "op = " << binopToStr(node.op()) << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "\"";
}

void GraphvizPrinter::printCustomNode(const CustomNode& node) {
  os() << "label=\"" << node.name() << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "\"";
}

void GraphvizPrinter::printIndexNode(const IndexNode& node) {
  os() << "label=\""
       << "IndexNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "indices = ";
  printIndices(node.indices()) << "\\n"
                               << "\"";
}

void GraphvizPrinter::printIndexedUpdateNode(const IndexedUpdateNode& node) {
  os() << "label=\""
       << "IndexedUpdateNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "indices = {\\n";
  for (const auto& indices : node.indexings()) {
    printIndices(indices) << "\\n";
  }
  os() << "}\"";
}

void GraphvizPrinter::printRangeIndex(const range& rangeIdx) {
  os() << rangeIdx.start() << ":";
  if (rangeIdx.end().has_value()) {
    os() << rangeIdx.end().value();
  }
  os() << ":" << rangeIdx.stride();
}

std::ostream& GraphvizPrinter::printIndices(const std::vector<Index>& indices) {
  os() << "[";
  for (unsigned i = 0; i < indices.size(); i++) {
    const auto& idx = indices[i];
    switch (idx.type()) {
      case detail::IndexType::Literal:
        os() << idx.get<Dim>();
        break;
      case detail::IndexType::Span:
        os() << ":";
        break;
      case detail::IndexType::Range:
        printRangeIndex(idx.get<range>());
        break;
      case detail::IndexType::Tensor:
        os() << "<Tensor>";
        break;
    }
    if (i != indices.size() - 1) {
      os() << ", ";
    }
  }
  return os();
}

std::ostream& GraphvizPrinter::printScalarValue(const ScalarNode& node) {
  switch (node.dataType()) {
    case dtype::f16:
      throw std::runtime_error(
          "[GraphvizPrinter::printScalarNodeValue] f16 is unsupported");
    case dtype::f32:
      os() << node.scalar<float>();
      break;
    case dtype::f64:
      os() << node.scalar<double>();
      break;
    case dtype::b8:
      os() << node.scalar<char>();
      break;
    case dtype::s16:
      os() << node.scalar<short>();
      break;
    case dtype::s32:
      os() << node.scalar<int>();
      break;
    case dtype::s64:
      os() << node.scalar<long long>();
      break;
    case dtype::u8:
      os() << node.scalar<unsigned char>();
      break;
    case dtype::u16:
      os() << node.scalar<unsigned short>();
      break;
    case dtype::u32:
      os() << node.scalar<unsigned int>();
      break;
    case dtype::u64:
      os() << node.scalar<unsigned long long>();
      break;
    default:
      throw std::runtime_error("Unknown data type");
  }
  return os();
}

void GraphvizPrinter::printScalarNode(const ScalarNode& node) {
  os() << "label=\""
       << "ScalarNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "dtype = " << node.dataType() << "\\n"
       << "value = ";
  printScalarValue(node) << "\\n"
                         << "\"";
}

void GraphvizPrinter::printValueNode(const ValueNode& node) {
  os() << "label=\""
       << "ValueNode"
       << "\\n"
       << "shape = " << node.shape() << "\\n"
       << "\"";
}

std::ostream& GraphvizPrinter::printNodes(NodePtr node) {
  if (!nodeNamer_.contains(node)) {
    // roots at bottom
    for (const auto& input : node->inputs()) {
      printNodes(input);
    }
    os() << "    " << nodeNamer_.getName(node);
    os() << "  ["; // node attributes
    os() << "color=\"" << nodeColor_ << "\" ";
    switch (node->type()) {
      case NodeType::Binary:
        printBinaryNode(node->impl<BinaryNode>());
        break;
      case NodeType::Custom:
        printCustomNode(node->impl<CustomNode>());
        break;
      case NodeType::Index:
        printIndexNode(node->impl<IndexNode>());
        break;
      case NodeType::IndexedUpdate:
        printIndexedUpdateNode(node->impl<IndexedUpdateNode>());
        break;
      case NodeType::Scalar:
        printScalarNode(node->impl<ScalarNode>());
        break;
      case NodeType::Value:
        printValueNode(node->impl<ValueNode>());
        break;
    }
    os() << "];" << std::endl;
  }
  return os();
}

std::ostream& GraphvizPrinter::printEdges(NodePtr node) {
  if (edgesPrinted_.find(node) == edgesPrinted_.end()) {
    edgesPrinted_.insert(node);
    // root at bottom
    for (const auto& input : node->inputs()) {
      printEdges(input);
    }
    const auto& nodeName = nodeNamer_.getName(node);
    for (const auto& input : node->inputs()) {
      os() << "    " << nodeName << " -> " << nodeNamer_.getName(input);
      os() << " ["; // edge attributes
      os() << "color=\"" << edgeColor_ << "\"";
      os() << " ]" << std::endl;
    }
  }
  return os();
}

GraphvizPrinter::GraphvizPrinter(std::ostream& os) : os_(os), nodeNamer_("n") {
  this->os() << "digraph G {\n"
             << "rankdir=BT\n\n"; // root nodes at bottom
}

GraphvizPrinter::~GraphvizPrinter() {
  this->os() << "}\n";
}

std::ostream& operator<<(
    std::ostream& os,
    const GraphvizPrinter::Color& color) {
  switch (color) {
    case GraphvizPrinter::Color::Black:
      return os << "black";
    case GraphvizPrinter::Color::Green:
      return os << "green";
    case GraphvizPrinter::Color::Red:
      return os << "red";
  }
  throw std::runtime_error(
      "[operator<< for GraphvizPrinter::Color] unknown color");
}

GraphvizPrinter& GraphvizPrinter::printSubgraph(
    NodePtr node,
    const std::string& namePrefix) {
  const auto prefix = "cluster_" + namePrefix; // "cluster_" adds boundary
  os() << "subgraph " << subgraphNameGenerator_.genName(prefix) << " {\n";
  printNodes(node) << std::endl;
  printEdges(node);
  os() << "}\n";
  return *this;
}

} // namespace fl
