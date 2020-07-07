/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/common/System.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <array>
#include <cstdlib>
#include <ctime>
#include <functional>

#include "libraries/common/String.h"

namespace fl {
namespace lib {

std::string pathsConcat(const std::string& p1, const std::string& p2) {
  char sep = '/';

#ifdef _WIN32
  sep = '\\';
#endif

  if (!p1.empty() && p1[p1.length() - 1] != sep) {
    return (trim(p1) + sep + trim(p2)); // Need to add a path separator
  } else {
    return (trim(p1) + trim(p2));
  }
}

bool dirExists(const std::string& path) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    return false;
  } else if (info.st_mode & S_IFDIR) {
    return true;
  } else {
    return false;
  }
}

void dirCreate(const std::string& path) {
  if (dirExists(path)) {
    return;
  }
  mode_t nMode = 0755;
  int nError = 0;
#ifdef _WIN32
  nError = _mkdir(path.c_str());
#else
  nError = mkdir(path.c_str(), nMode);
#endif
  if (nError != 0) {
    throw std::runtime_error(
        std::string() + "Unable to create directory - " + path);
  }
}

bool fileExists(const std::string& path) {
  std::ifstream fs(path, std::ifstream::in);
  return fs.good();
}

std::string getEnvVar(
    const std::string& key,
    const std::string& dflt /*= "" */) {
  char* val = getenv(key.c_str());
  return val ? std::string(val) : dflt;
}

std::string getCurrentDate() {
  time_t now = time(nullptr);
  struct tm tmbuf;
  struct tm* tstruct;
  tstruct = localtime_r(&now, &tmbuf);

  std::array<char, 80> buf;
  strftime(buf.data(), buf.size(), "%Y-%m-%d", tstruct);
  return std::string(buf.data());
}

std::string getCurrentTime() {
  time_t now = time(nullptr);
  struct tm tmbuf;
  struct tm* tstruct;
  tstruct = localtime_r(&now, &tmbuf);

  std::array<char, 80> buf;
  strftime(buf.data(), buf.size(), "%X", tstruct);
  return std::string(buf.data());
}

std::vector<std::string> getFileContent(const std::string& file) {
  std::vector<std::string> data;
  std::ifstream in = createInputStream(file);
  std::string str;
  while (std::getline(in, str)) {
    data.emplace_back(str);
  }
  in.close();
  return data;
}

std::ifstream createInputStream(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  return file;
}

std::ofstream createOutputStream(
    const std::string& filename,
    std::ios_base::openmode mode) {
  std::ofstream file(filename, mode);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  return file;
}

} // namespace lib
} // namespace fl