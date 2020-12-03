/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

namespace fl {
namespace lib {
namespace text {

/**
 * PartialFileReader is a text file reader designed to run in distributed
 * manner, where each reader focuses on reading only part of the file. The text
 * file will be split into `totalReaders` parts with roughly equal size and each
 * single part ends in a completed sentence. Each reader will then focus on
 * reading the `rank`th part of the file.
 *
 * Note that this reader supports sequential reading only.
 *
 * Usage:
 *
 * # In worker `rank`.
 * PartialFileReader reader(rank, totalReaders);
 * for (file in files) {
 *   reader.loadFile(file);
 *   while (reader.hasNextLine()) {
 *     std::string text = reader.getLine();
 *   }
 * }
 */

class PartialFileReader {
 public:
  PartialFileReader(int rank, int totalReaders);

  // Explicitly disabling copy constructor and copy assignment operator to make
  // sure unique access to the partial reading stream.
  PartialFileReader(const PartialFileReader&) = delete;
  PartialFileReader& operator=(const PartialFileReader&) = delete;

  void loadFile(const std::string& filename);

  size_t getPosition();
  bool hasNextLine();

  std::string getLine();
  std::vector<std::string> getLines();

  int getRank();
  int getTotalReaders();

 private:
  int rank_;
  int totalReaders_;

  // Each reader maintains a single stream to read from, so as neither to open
  // the same file for many times nor to read the whole text into memory at
  // once.
  std::ifstream stream_;

  // The ending position of current stream. `hasNextLine()` returns false when
  // reaching this point.
  size_t end_;
};

} // namespace text
} // namespace lib
} // namespace fl
