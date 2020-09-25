/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests logging command line flags. You may run in this manner:
// flashlight/test/LoggingFlagsTest --fl_vlog_level=5 --fl_log_level=ERROR

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/flashlight/common/Logging.h"

using namespace fl;

namespace {

using testing::Eq;
using testing::HasSubstr;
using testing::Not;

TEST(Logging, logFlag) {
  std::vector<LogLevel> levels = {fl::INFO, fl::WARNING, fl::ERROR};
  for (LogLevel level : levels) {
    std::stringstream ss;
    ss << "log-level-" << logLevelName(level);
    const std::string logMsg = ss.str();
    FL_LOG(level) << logMsg;
  }
}

TEST(LoggingDeathTest, logFlag) {
  LogLevel level = fl::FATAL;
  std::stringstream ss;
  ss << "log-level-" << logLevelName(level);
  const std::string logMsg = ss.str();
  EXPECT_DEATH((FL_LOG(level) << logMsg), ss.str());
}

TEST(Logging, vlogFlag) {
  for (int i = 0; i < 15; ++i) {
    std::stringstream ss;
    ss << "vlog-level-" << i;
    const std::string logMsg = ss.str();
    FL_VLOG(i) << logMsg;
  }
}

} // namespace

int main(int argc, char** argv) {
  fl::initFlLogging(argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
