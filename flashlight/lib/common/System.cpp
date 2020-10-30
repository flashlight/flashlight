/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/common/System.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <array>
#include <cstdlib>
#include <ctime>
#include <functional>

#include "flashlight/lib/common/String.h"

namespace fl {
namespace lib {

std::string pathSeperator() {
#ifdef _WIN32
  return "\\";
#else
  return "/";
#endif
}

std::string pathsConcat(const std::string& p1, const std::string& p2) {
  if (!p1.empty() && p1[p1.length() - 1] != pathSeperator()[0]) {
    return (
        trim(p1) + pathSeperator() + trim(p2)); // Need to add a path separator
  } else {
    return (trim(p1) + trim(p2));
  }
}

std::vector<std::string> getDirsOnPath(const std::string& path) {
  const std::string path_ = trim(path);

  if (path_.empty() || path_ == pathSeperator() || path_ == "." ||
      path_ == "..") {
    return {path_};
  }
  const std::vector<std::string> tokens = split(pathSeperator(), path_);
  std::vector<std::string> dirs;
  for (const std::string& token : tokens) {
    const std::string dir = trim(token);
    if (!dir.empty()) {
      dirs.push_back(dir);
    }
  }
  return dirs;
}

std::string dirname(const std::string& path) {
  std::vector<std::string> dirsOnPath = getDirsOnPath(path);
  if (dirsOnPath.size() < 2) {
    return ".";
  } else {
    dirsOnPath.pop_back();
    const std::string root =
        ((trim(path))[0] == pathSeperator()[0]) ? pathSeperator() : "";
    return root + join(pathSeperator(), dirsOnPath);
  }
}

std::string basename(const std::string& path) {
  std::vector<std::string> dirsOnPath = getDirsOnPath(path);
  if (dirsOnPath.empty()) {
    return "";
  } else {
    return dirsOnPath.back();
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

void dirCreateRecursive(const std::string& path) {
  if (dirExists(path)) {
    return;
  }
  std::vector<std::string> dirsOnPath = getDirsOnPath(path);
  std::string pathFromStart;
  if (path[0] == pathSeperator()[0]) {
    pathFromStart = pathSeperator();
  }
  for (std::string& dir : dirsOnPath) {
    if (pathFromStart.empty()) {
      pathFromStart = dir;
    } else {
      pathFromStart = pathsConcat(pathFromStart, dir);
    }

    if (!dirExists(pathFromStart)) {
      dirCreate(pathFromStart);
    }
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
