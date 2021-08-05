/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/common/Logging.h"

using namespace fl;

namespace {

using testing::HasSubstr;
using testing::Not;

// FL_VLOG(l) should print to stdout when VerboseLogging::setMaxLoggingLevel(i)
// i>=l
TEST(Logging, vlogOnOff) {
  std::stringstream stdoutBuffer;
  std::stringstream stderrBuffer;

  std::streambuf* origStdoutBuffer = std::cout.rdbuf();
  std::streambuf* origStderrBuffer = std::cerr.rdbuf();

  std::cout.rdbuf(stdoutBuffer.rdbuf());
  std::cerr.rdbuf(stderrBuffer.rdbuf());

  for (int i = 0; i < 11; ++i) {
    stdoutBuffer.clear();
    stderrBuffer.clear();
    VerboseLogging::setMaxLoggingLevel(i);
    FL_VLOG(0) << "vlog-0";
    FL_VLOG(1) << "vlog-1";
    FL_VLOG(10) << "vlog-10";

    // Prints to stderr
    EXPECT_THAT(stderrBuffer.str(), HasSubstr("vlog-0"));

    if (i >= 1) {
      EXPECT_THAT(stderrBuffer.str(), HasSubstr("vlog-1"));
    } else {
      EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("vlog-1")));
    }

    if (i >= 10) {
      EXPECT_THAT(stderrBuffer.str(), HasSubstr("vlog-10"));
    } else {
      EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("vlog-10")));
    }

    // Does not print to stdout
    EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("vlog-0")));
    EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("vlog-1")));
    EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("vlog-10")));
  }

  std::cout.rdbuf(origStdoutBuffer);
  std::cerr.rdbuf(origStderrBuffer);
}

// FL_LOG(l) should print to stdout when Logging::setMaxLoggingLevel(i) i>=l and
// l is fl::INFO or WARNING.
// FL_LOG(l) should print to stderr when Logging::setMaxLoggingLevel(i) i>=l and
// l is ERROR.
TEST(Logging, logOnOff) {
  std::stringstream stdoutBuffer;
  std::stringstream stderrBuffer;

  std::streambuf* origStdoutBuffer = std::cout.rdbuf();
  std::streambuf* origStderrBuffer = std::cerr.rdbuf();

  std::cout.rdbuf(stdoutBuffer.rdbuf());
  std::cerr.rdbuf(stderrBuffer.rdbuf());

  const std::vector<LogLevel> logLevels = {
      fl::DISABLED, fl::FATAL, fl::ERROR, fl::WARNING, fl::INFO};
  for (LogLevel l : logLevels) {
    stdoutBuffer.clear();
    stderrBuffer.clear();

    Logging::setMaxLoggingLevel(l);
    FL_LOG(fl::INFO) << "log-info";
    FL_LOG(WARNING) << "log-warning";
    FL_LOG(ERROR) << "log-error";

    // Prints to stderr
    if (l >= fl::INFO) {
      EXPECT_THAT(stderrBuffer.str(), HasSubstr("log-info"));
    } else {
      EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("log-info")));
    }

    if (l >= WARNING) {
      EXPECT_THAT(stderrBuffer.str(), HasSubstr("log-warning"));
    } else {
      EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("log-warning")));
    }

    // Does not print to stdout
    EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("log-info")));
    EXPECT_THAT(stdoutBuffer.str(), Not(HasSubstr("log-warning")));

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
  std::stringstream stderrBuffer;
  std::streambuf* origStderrBuffer = std::cerr.rdbuf();
  std::cerr.rdbuf(stderrBuffer.rdbuf());

  Logging::setMaxLoggingLevel(DISABLED);
  FL_LOG(fl::FATAL) << "log-fatal";
  EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("log-fatal")));
  EXPECT_THAT(stderrBuffer.str(), Not(HasSubstr("log-fatal")));

  std::cerr.rdbuf(origStderrBuffer);

  Logging::setMaxLoggingLevel(fl::FATAL);
  EXPECT_DEATH({ FL_LOG(fl::FATAL) << "log-fatal"; }, "");
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
