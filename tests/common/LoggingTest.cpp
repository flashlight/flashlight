/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/common/Logging.h"

using namespace fl;

namespace {

using testing::HasSubstr;
using testing::Not;

// VLOG(l) should print to stdout when VerboseLogging::setMaxLoggingLevel(i)
// i>=l
TEST(Logging, vlogOnOff) {
  LOG(INFO) << "test" << 1 << 1UL << 1L << 1.0;
  std::stringstream stdoutBuffer;
  std::stringstream stderrBuffer;

  std::streambuf* origStdoutBuffer = std::cout.rdbuf();
  std::streambuf* origStderrBuffer = std::cerr.rdbuf();

  std::cout.rdbuf(stdoutBuffer.rdbuf());
  std::cerr.rdbuf(stderrBuffer.rdbuf());

  for (int i = 0; i < 11; ++i) {
    stdoutBuffer.str(""); // clear content
    stderrBuffer.str(""); // clear content
    VerboseLogging::setMaxLoggingLevel(i);
    VLOG(0) << "vlog-0";
    VLOG(1) << "vlog-1";
    VLOG(10) << "vlog-10";

    // Prints to stdout
    EXPECT_THAT(stdoutBuffer.str(), HasSubstr("vlog-0"));

    if (i >= 1) {
      EXPECT_THAT(stdoutBuffer.str(), HasSubstr("vlog-1"));
    } else {
      EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("vlog-1")));
    }

    if (i >= 10) {
      EXPECT_THAT(stdoutBuffer.str(), HasSubstr("vlog-10"));
    } else {
      EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("vlog-10")));
    }

    // Does not print to stderr
    EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("vlog-0")));
    EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("vlog-1")));
    EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("vlog-10")));
  }

  std::cout.rdbuf(origStdoutBuffer);
  std::cerr.rdbuf(origStderrBuffer);
}

// LOG(l) should print to stdout when Logging::setMaxLoggingLevel(i) i>=l and l
// is INFO or WARNING.
// LOG(l) should print to stderr when Logging::setMaxLoggingLevel(i) i>=l and l
// is ERROR.
TEST(Logging, logOnOff) {
  std::stringstream stdoutBuffer;
  std::stringstream stderrBuffer;

  std::streambuf* origStdoutBuffer = std::cout.rdbuf();
  std::streambuf* origStderrBuffer = std::cerr.rdbuf();

  std::cout.rdbuf(stdoutBuffer.rdbuf());
  std::cerr.rdbuf(stderrBuffer.rdbuf());

  const std::vector<LogLevel> logLevels = {
      DISABLE_LOGGING, FATAL, ERROR, WARNING, INFO};
  for (LogLevel l : logLevels) {
    stdoutBuffer.str(""); // clear content
    stderrBuffer.str(""); // clear content

    Logging::setMaxLoggingLevel(l);
    LOG(INFO) << "log-info";
    LOG(WARNING) << "log-warning";
    LOG(ERROR) << "log-error";

    // Prints to stdout
    if (l >= INFO) {
      EXPECT_THAT(stdoutBuffer.str(), HasSubstr("log-info"));
    } else {
      EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("log-info")));
    }

    if (l >= WARNING) {
      EXPECT_THAT(stdoutBuffer.str(), HasSubstr("log-warning"));
    } else {
      EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("log-warning")));
    }

    EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("log-error")));

    // Prints to stderr
    EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("log-info")));
    EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("log-warning")));

    if (l >= ERROR) {
      EXPECT_THAT(stderrBuffer.str(), HasSubstr("log-error"));
    } else {
      EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("log-error")));
    }
  }

  std::cout.rdbuf(origStdoutBuffer);
  std::cerr.rdbuf(origStderrBuffer);
}

TEST(LoggingDeathTest, FatalOnOff) {
  std::stringstream stdoutBuffer;
  std::stringstream stderrBuffer;

  std::streambuf* origStdoutBuffer = std::cout.rdbuf();
  std::streambuf* origStderrBuffer = std::cerr.rdbuf();

  std::cout.rdbuf(stdoutBuffer.rdbuf());
  std::cerr.rdbuf(stderrBuffer.rdbuf());

  Logging::setMaxLoggingLevel(DISABLE_LOGGING);
  LOG(FATAL) << "log-fatal";
  EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("log-fatal")));
  EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("log-fatal")));

  std::cout.rdbuf(origStdoutBuffer);
  std::cerr.rdbuf(origStderrBuffer);

  Logging::setMaxLoggingLevel(FATAL);
  EXPECT_DEATH({ LOG(FATAL) << "log-fatal"; }, "");
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
