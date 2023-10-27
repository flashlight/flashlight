/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <unordered_set>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexedUpdateNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

namespace detail {

class NameGenerator {
  unsigned namemCounter_;

 public:
  NameGenerator();
  std::string genName(std::string prefix);
};

class NodeNamer {
  const std::string prefix_;
  NameGenerator nodeNameGenerator_;
  std::unordered_map<NodePtr, std::string> nodeToName_;

 public:
  NodeNamer(const std::string& defaultPrefix);
  const std::string& getName(NodePtr node);
  bool contains(NodePtr node) const;
};

} // namespace detail

/**
 * A printer that prints a computation graph to GraphViz DOT format.
 */
class GraphvizPrinter {
 public:
  enum class Color {
    Black,
    Green,
    Red,
  };

 private:
  std::ostream& os_;
  detail::NameGenerator subgraphNameGenerator_;
  detail::NodeNamer nodeNamer_;
  std::unordered_set<NodePtr> edgesPrinted_;
  // temporary state use when printing a single subgraph
  const std::unordered_map<NodePtr, float>* nodeToTotTimeMs_{};
  float medianTotTime_ = 0;
  float maxTotTime_ = 0;

  Color edgeColor_ = Color::Black;
  Color nodeColor_ = Color::Black;

  std::ostream& os();
  void printBinaryNodeLabels(const BinaryNode& node);
  void printCustomNodeLabels(const CustomNode& node);
  void printIndexNodeLabels(const IndexNode& node);
  void printIndexedUpdateNodeLabels(const IndexedUpdateNode& node);
  std::ostream& printIndices(const std::vector<Index>& indices);
  void printRangeIndex(const range& rangeIdx);
  void printScalarNodeLabels(const ScalarNode& node);
  std::ostream& printScalarValue(const ScalarNode& node);
  void printValueNodeLabels(const ValueNode& node);
  std::ostream& printNodes(NodePtr node);
  std::ostream& printNodeLabels(NodePtr node);
  std::ostream& printNodeColor(float tottime);
  std::ostream& printRelativeColor(float tottime);
  std::ostream& printEdges(NodePtr node);

  GraphvizPrinter& printSubgraphImpl(
      NodePtr node,
      const std::string& namePrefix);

 public:
  // no copy/move
  GraphvizPrinter(std::ostream& os); // wrap
  ~GraphvizPrinter(); // wrap
  GraphvizPrinter(const GraphvizPrinter&) = delete;
  GraphvizPrinter(GraphvizPrinter&&) = delete;
  GraphvizPrinter& operator=(const GraphvizPrinter&) = delete;
  GraphvizPrinter& operator=(GraphvizPrinter&&) = delete;

  /**
   * Applies to all newly printed edges after this function call
   * TODO
   */
  GraphvizPrinter& setEdgeColor(Color newColor);

  /**
   * Applies to all newly printed nodes after this function call
   * TODO
   */
  GraphvizPrinter& setNodeColor(Color newColor);

  /**
   * Print the entire computation tree rooted at `node` to given stream as a
   * `subgraph` DOT construct. This way, user can visually differentiate graphs
   * from different materializations, if they print out a subgraph for each
   * materialization.
   *
   * TODO
   */
  GraphvizPrinter& printSubgraph(NodePtr node, const std::string& namePrefix);

  // color node with execution statistics
  GraphvizPrinter& printSubgraph(
      NodePtr node,
      const std::string& namePrefix,
      const std::unordered_map<NodePtr, float>& nodeToTotTimeMs_);
};

std::ostream& operator<<(std::ostream& os, const GraphvizPrinter::Color& color);

} // namespace fl
