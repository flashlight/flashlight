/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sstream>

#include "flashlight/app/lm/common/Defines.h"
#include "flashlight/pkg/runtime/Runtime.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/tokenizer/Tokenizer.h"

/**
 * Build dictioinary for LM training
 *
 * Usage:
 *
 *  dictionary_builder \
 *   --data_dir=/tmp \
 *   --data_train=test1.txt,test2.txt \
 *   --n_workers=40 \
 *   --dictionary=dictionary.txt \
 *   --dictionary_min_appearence=2 \
 *   --dictionary_max_size=200000 \
 *   --write_meta=true
 *
 * -------------------------------
 *
 * It reads multiple files and count the tokens in them. Tokens will first be
 * filter by appearance and then be saved out as dictionary. If `write_meta` is
 * on, the meta data of each training file will also be generated in the same
 * folder with suffix `.desc`.
 */

namespace {
DEFINE_string(
    data_dir,
    "",
    "Prefix for the 'data_train' and 'data_valid' files.");
DEFINE_string(
    data_train,
    "",
    "Comma-separated list of training data files; '--data_dir' will be used to add prefix for the files.");

DEFINE_string(
    dictionary,
    "",
    "Path to the dictionary file (read/write), which defines tokens set of language model.");
DEFINE_int64(
    dictionary_max_size,
    -1,
    "Number of rows to use from the dictionary file (top rows), cutting the number of target classes.");
DEFINE_int64(
    dictionary_min_appearence,
    0,
    "Minimum occurence of a token which is allowed in the dictionary");

DEFINE_int64(n_workers, 1, "Number of workers for parallel file reading");
DEFINE_bool(
    write_meta,
    false,
    "Generate (true) or not (false) the meta data of a file");
} // namespace

int main(int argc, char** argv) {
  fl::init();
  std::string exec(argv[0]);
  gflags::SetUsageMessage(
      "Preparation of tokens dictionary from the text data. \n Usage: " + exec +
      " \n Compulsory: [--data_train] [--dictionary]");
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  LOG(INFO) << "Gflags after parsing \n" << fl::pkg::runtime::serializeGflags("; ");

  if (argc <= 1 || FLAGS_data_train.empty() || FLAGS_dictionary.empty()) {
    throw std::invalid_argument(gflags::ProgramUsage());
  }

  auto tokenizer = fl::lib::text::Tokenizer();
  auto files = fl::lib::split(',', FLAGS_data_train);

  for (const auto& file : files) {
    LOG(INFO) << "Parsing " << file;
    tokenizer.countTokens(
        fl::lib::pathsConcat(FLAGS_data_dir, file),
        FLAGS_n_workers,
        FLAGS_write_meta);

    if (FLAGS_write_meta) {
      auto metaPath = file + ".desc";

      auto stream = fl::lib::createOutputStream(metaPath);
      auto fileMetaData = tokenizer.getTextFileMetaData();
      for (int i = 0; i < fileMetaData.size(); ++i) {
        stream << fileMetaData[i].first << " " << fileMetaData[i].second
               << "\n";
      }
      LOG(INFO) << "  Meta data saved to: " << metaPath;
    }
  }

  LOG(INFO) << " --- Data Loading completed. --- ";
  LOG(INFO) << "  Loaded " << tokenizer.totalTokens() << " tokens";
  LOG(INFO) << "  Loaded " << tokenizer.totalSentences() << " sentences";

  tokenizer.pruneTokens(
      FLAGS_dictionary_max_size, FLAGS_dictionary_min_appearence);
  auto tokenCountPairs = tokenizer.getDictionary();
  auto stream = fl::lib::createOutputStream(FLAGS_dictionary);
  stream << fl::lib::text::kEosToken << " 0\n";
  stream << fl::lib::text::kUnkToken << " 0\n";
  stream << fl::lib::text::kPadToken << " 0\n";
  stream << fl::lib::text::kMaskToken << " 0\n";
  for (const auto& tcp : tokenCountPairs) {
    stream << tcp.first << " " << tcp.second << "\n";
  }

  LOG(INFO) << "Dictionary saved to: " << FLAGS_dictionary;

  return 0;
}
