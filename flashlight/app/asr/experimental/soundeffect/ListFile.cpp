/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dirent.h>
#include <sys/types.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using ::fl::lib::dirCreateRecursive;
using ::fl::lib::dirname;
using ::fl::lib::join;
using ::fl::lib::pathsConcat;
using ::fl::lib::split;
using ::fl::lib::splitOnWhitespace;
using ::fl::lib::trim;

namespace fl {
namespace app {
namespace asr {
namespace sfx {

void ListFileEntry::serialize(std::ostream& os) const {
  os << sampleId_ << ' ' << audioFilePath_ << ' ' << audioSize_ << ' '
     << join(" ", transcript_);
}

ListFileEntry ListFileEntry::deserialize(std::istream& is) {
  std::string line;
  std::getline(is, line);
  std::vector<std::string> tokens = splitOnWhitespace(line, true);
  if (tokens.size() < 3) {
    std::stringstream ss;
    ss << "ListFileEntry::deserialize() invalid entry."
       << " Minimum of 3 tokens required but found=" << tokens.size()
       << " line={" << line << '}';
    throw std::runtime_error(ss.str());
  }

  ListFileEntry entry;
  entry.sampleId_ = std::move(tokens[0]);
  entry.audioFilePath_ = std::move(tokens[1]);
  entry.audioSize_ = std::stoul(tokens[2]);
  entry.transcript_.resize(tokens.size() - 3);

  for (int i = 3; i < tokens.size(); ++i) {
    entry.transcript_[i - 3] = std::move(tokens[i]);
  }

  return entry;
}

std::string ListFileEntry::toString() const {
  std::stringstream ss;
  serialize(ss);
  return ss.str();
}

std::string ListFileEntry::prettyString() const {
  std::stringstream ss;
  ss << "sampleId_=" << sampleId_ << " audioFilePath_=" << audioFilePath_
     << " audioSize_=" << audioSize_ << " transcript_={"
     << join(" ", transcript_) << '}';
  return ss.str();
}

ListFileReader::ListFileReader(
    const std::string& commaSeparatedListFilenames,
    const std::string& rootPath)
    : commaSeparatedListFilenames_(commaSeparatedListFilenames),
      rootPath_(rootPath) {
  readAllListFiles();
}

void ListFileReader::readAllListFiles() {
  std::vector<std::string> filesVec = split(',', commaSeparatedListFilenames_);
  for (const std::string& filename : filesVec) {
    const std::string fullpath = pathsConcat(rootPath_, trim(filename));
    readListFile(fullpath);
  }
}

std::string ListFileReader::prettyString() const {
  std::stringstream ss;
  ss << "listFileEntryVec_.size()=" << listFileEntryVec_.size()
     << " rootPath_=" << rootPath_
     << " commaSeparatedListFilenames_=" << commaSeparatedListFilenames_;
  return ss.str();
}

void ListFileReader::readListFile(const std::string& listFilePath) {
  std::ifstream listFile(listFilePath);
  if (!listFile) {
    std::stringstream ss;
    ss << "ListFileReader::readListFile(listFilePath='" << listFilePath
       << "') failed to open the file.";
    throw std::runtime_error(ss.str());
  }

  for (int line = 1;; ++line) {
    try {
      listFileEntryVec_.push_back(ListFileEntry::deserialize(listFile));
    } catch (std::exception& ex) {
      if (listFile.eof()) {
        break;
      }
      FL_LOG(fl::ERROR) << "ListFileReader::readListFile(listFilePath='"
                        << listFilePath << "') failed read line number=" << line
                        << " with error={" << ex.what() << '}';
    }
  }
}

size_t ListFileReader::size() const {
  return listFileEntryVec_.size();
}

const ListFileEntry& ListFileReader::read(int index) const {
  return listFileEntryVec_.at(index);
}

ListFileWriter::ListFileWriter(
    const std::string& listFileFilename,
    const std::string& rootPath)
    : listFileFilename_(listFileFilename),
      rootPath_(rootPath),
      numberOfSamples_(0) {}

ListFileWriter::~ListFileWriter() {
  listFile_.flush();
}

std::string ListFileWriter::prettyString() const {
  std::stringstream ss;
  ss << "listFile_.good()=" << listFile_.good()
     << " numberOfSamples_=" << numberOfSamples_ << " rootPath_=" << rootPath_
     << " listFileFilename_=" << listFileFilename_;
  return ss.str();
}

size_t ListFileWriter::size() const {
  return numberOfSamples_;
}

void ListFileWriter::write(const ListFileEntry& entry) {
  if (!listFile_.is_open()) {
    // Create directory
    const std::string fullPath =
        pathsConcat(rootPath_, trim(listFileFilename_));
    try {
      const std::string path = dirname(fullPath);
      dirCreateRecursive(path);
    } catch (std::exception& ex1) {
      std::stringstream ss;
      ss << "ListFileWriter::write(entry) this={" << prettyString()
         << "} with error={" << ex1.what() << "}";
      throw std::runtime_error(ss.str());
    }

    // Open file
    try {
      listFile_.open(fullPath.c_str(), std::ofstream::out | std::ofstream::app);
    } catch (std::exception& ex2) {
      std::stringstream ss;
      ss << "ListFileWriter::write(entry={" << entry.prettyString()
         << "} failed to open fullPath=" << fullPath << " this={"
         << prettyString() << "} with error={" << ex2.what() << "}";
      throw std::runtime_error(ss.str());
    }
  }
  entry.serialize(listFile_);
  listFile_ << std::endl;
  ++numberOfSamples_;
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
