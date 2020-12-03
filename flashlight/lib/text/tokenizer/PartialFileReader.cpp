/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/text/tokenizer/PartialFileReader.h"

namespace fl {
namespace lib {
namespace text {

PartialFileReader::PartialFileReader(int rank, int totalReaders)
    : rank_(rank), totalReaders_(totalReaders) {
  if (rank_ < 0 || rank_ > totalReaders_) {
    throw std::invalid_argument(
        "Invalid rank: " + std::to_string(rank_) + ", given " +
        std::to_string(totalReaders_) + " readers in total.");
  }
}

void PartialFileReader::loadFile(const std::string& filename) {
  stream_.close();
  stream_ = createInputStream(filename);
  stream_.seekg(0, stream_.end);
  const size_t fileSize = stream_.tellg();

  // Select the ending point
  end_ = fileSize;
  const size_t chunkSize = fileSize / totalReaders_;
  std::string line;
  if (rank_ < totalReaders_ - 1) {
    stream_.seekg(chunkSize * (rank_ + 1), std::ios::beg);
    std::getline(stream_, line);
    end_ = stream_.tellg();
  }

  // Set stream_ to its starting point
  stream_.seekg(chunkSize * rank_, std::ios::beg);
  if (rank_ > 0) {
    std::getline(stream_, line);
  }
}

size_t PartialFileReader::getPosition() {
  return stream_.tellg();
}

bool PartialFileReader::hasNextLine() {
  return getPosition() < end_;
}

std::string PartialFileReader::getLine() {
  std::string line;
  std::getline(stream_, line);
  return trim(line);
}

std::vector<std::string> PartialFileReader::getLines() {
  std::vector<std::string> lines;
  while (hasNextLine()) {
    lines.emplace_back(getLine());
  }
  return lines;
}

int PartialFileReader::getRank() {
  return rank_;
}

int PartialFileReader::getTotalReaders() {
  return totalReaders_;
}

} // namespace text
} // namespace lib
} // namespace fl
