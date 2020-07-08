/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sstream>

#include "common/Defines.h"

#include "libraries/language/tokenizer/Tokenizer.h"
#include "libraries/common/String.h"
#include "libraries/common/System.h"

namespace {
DEFINE_int64(nworkers, 1, "number of workers");
DEFINE_bool(
    writedescriptor,
    false,
    "if or not to generate the descriptor of a file");
} // namespace

using namespace fl::lib;
using namespace fl::task::lm;

std::string printGflags(const std::string& separator /* = "\n" */) {
  std::stringstream serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  gflags::GetAllFlags(&allFlags);
  std::string currVal;
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
    serialized << "--" << itr->name << "=" << currVal << separator;
  }
  return serialized.str();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  gflags::SetUsageMessage(
      "Usage: " + exec + " \n Compulsory: [-train] [-dictionary]" +
      "\n Optional: [nworkers] [nwords] [minappearence]");
  if (argc <= 1) {
    throw std::invalid_argument(gflags::ProgramUsage());
  }
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  LOG(INFO) << "Gflags after parsing \n" << printGflags("; ");

  auto tokenizer = Tokenizer();
  auto files = split(',', FLAGS_train);

  for (const auto& file : files) {
    LOG(INFO) << "Parsing " << file;
    tokenizer.countWords(file, FLAGS_nworkers, FLAGS_writedescriptor);
    LOG(INFO) << "  Loaded " << tokenizer.totalWords() << "  words";

    if (FLAGS_writedescriptor) {
      LOG(INFO) << "  Loaded " << tokenizer.totalSentences() << " sentences";
      auto descriptorPath = file + ".desc";
      tokenizer.saveFileDescriptor(descriptorPath);
      LOG(INFO) << "  Descriptor saved to: " << descriptorPath;
    }
  }

  if (!FLAGS_dictionary.empty()) {
    tokenizer.filterWords(FLAGS_maxwords, FLAGS_minappearence);
    tokenizer.saveDictionary(FLAGS_dictionary);
    LOG(INFO) << "Dictionary saved to: " << FLAGS_dictionary;
  }

  return 0;
}