/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace fl {
namespace lib {

size_t getProcessId();

size_t getThreadId();

std::string pathsConcat(const std::string& p1, const std::string& p2);

std::string pathSeperator();

std::string dirname(const std::string& path);

std::string basename(const std::string& path);

bool dirExists(const std::string& path);

void dirCreate(const std::string& path);

void dirCreateRecursive(const std::string& path);

bool fileExists(const std::string& path);

std::string getEnvVar(const std::string& key, const std::string& dflt = "");

std::string getCurrentDate();

std::string getCurrentTime();

std::string getTmpPath(const std::string& filename);

std::vector<std::string> getFileContent(const std::string& file);

std::vector<std::string> fileGlob(const std::string& pat);

std::ifstream createInputStream(const std::string& filename);

std::ofstream createOutputStream(
    const std::string& filename,
    std::ios_base::openmode mode = std::ios_base::out);

/**
 * Calls `f(args...)` repeatedly, retrying if an exception is thrown.
 * Supports sleeps between retries, with duration starting at `initial` and
 * multiplying by `factor` each retry. At most `maxIters` calls are made.
 */
template <class Fn, class... Args>
typename std::result_of<Fn(Args...)>::type retryWithBackoff(
    std::chrono::duration<double> initial,
    double factor,
    int64_t maxIters,
    Fn&& f,
    Args&&... args) {
  if (!(initial.count() >= 0.0)) {
    throw std::invalid_argument("retryWithBackoff: bad initial");
  } else if (!(factor >= 0.0)) {
    throw std::invalid_argument("retryWithBackoff: bad factor");
  } else if (maxIters <= 0) {
    throw std::invalid_argument("retryWithBackoff: bad maxIters");
  }
  auto sleepSecs = initial.count();
  for (int64_t i = 0; i < maxIters; ++i) {
    try {
      return f(std::forward<Args>(args)...);
    } catch (...) {
      if (i >= maxIters - 1) {
        throw;
      }
    }
    if (sleepSecs > 0.0) {
      /* sleep override */
      std::this_thread::sleep_for(
          std::chrono::duration<double>(std::min(1e7, sleepSecs)));
    }
    sleepSecs *= factor;
  }
  throw std::logic_error("retryWithBackoff: hit unreachable");
}
} // namespace lib
} // namespace fl
