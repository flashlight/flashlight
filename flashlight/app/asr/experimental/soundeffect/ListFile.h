/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "flashlight/app/asr/data/Utils.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

// The format of the list: columns should be space-separated
// [utterance id] [audio file (full path)] [audio length] [word transcripts]
struct ListFileEntry {
  std::string sampleId_; // utterance id
  std::string audioFilePath_; // full path to audio file
  size_t audioSize_ = 0;
  std::vector<std::string> transcript_; // word transcripts

  std::string toString() const;
  void serialize(std::ostream& os) const;
  // Reads the next line of is and parses it as a list file entry.
  static ListFileEntry deserialize(std::istream& is);
  std::string prettyString() const;
};

class ListFileReader {
 public:
  // @param commaSeparatedListFilenames string of comma separated filenames.
  // @param rootPath path to prefix to filenames that are relative paths.
  ListFileReader(
      const std::string& commaSeparatedListFilenames,
      const std::string& rootPath = "");

  size_t size() const;
  const ListFileEntry& read(int index) const;

  std::string prettyString() const;

 private:
  void readAllListFiles();
  void readListFile(const std::string& listFilePath);

  std::vector<ListFileEntry> listFileEntryVec_;
  const std::string commaSeparatedListFilenames_;
  const std::string rootPath_;
};

class ListFileWriter {
 public:
  // @param listFileFilename string of comma separated filenames.
  // @param rootPath path to prefix to filename if it is a relative path.
  ListFileWriter(
      const std::string& listFileFilename,
      const std::string& rootPath = "");
  ~ListFileWriter();

  size_t size() const;
  void write(const ListFileEntry& entry);

  std::string prettyString() const;

 private:
  const std::string listFileFilename_;
  const std::string rootPath_;
  std::ofstream listFile_;
  size_t numberOfSamples_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
