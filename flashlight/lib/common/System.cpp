/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/common/System.h"

#include <glob.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <array>
#include <cstdlib>
#include <ctime>
#include <functional>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "flashlight/lib/common/String.h"

namespace fl {
namespace lib {

size_t getProcessId() {
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return ::getpid();
#endif
}

size_t getThreadId() {
#ifdef _WIN32
  return GetCurrentThreadId();
#else
  return std::hash<std::thread::id>()(std::this_thread::get_id());
#endif
}

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

namespace {

/**
 * @path contains directories separated by path separator.
 * Returns a vector with the directores in the original order. Vector with a
 * Special cases: a vector with a single entry containing the input is returned
 * when path is one of the following special cases: empty, “.”, “..” and “/”
 */
std::vector<std::string> getDirsOnPath(const std::string& path) {
  const std::string trimPath = trim(path);

  if (trimPath.empty() || trimPath == pathSeperator() || trimPath == "." ||
      trimPath == "..") {
    return {trimPath};
  }
  const std::vector<std::string> tokens = split(pathSeperator(), trimPath);
  std::vector<std::string> dirs;
  for (const std::string& token : tokens) {
    const std::string dir = trim(token);
    if (!dir.empty()) {
      dirs.push_back(dir);
    }
  }
  return dirs;
}

} // namespace

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

std::string getTmpPath(const std::string& filename) {
  std::string tmpDir = "/tmp";
  auto getTmpDir = [&tmpDir](const std::string& env) {
    char* dir = std::getenv(env.c_str());
    if (dir != nullptr) {
      tmpDir = std::string(dir);
    }
  };
  getTmpDir("TMPDIR");
  getTmpDir("TEMP");
  getTmpDir("TMP");
  return tmpDir + "/fl_tmp_" + getEnvVar("USER", "unknown") + "_" + filename;
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

std::vector<std::string> fileGlob(const std::string& pat) {
  glob_t result;
  glob(pat.c_str(), GLOB_TILDE, nullptr, &result);
  std::vector<std::string> ret;
  for (unsigned int i = 0; i < result.gl_pathc; ++i) {
    ret.push_back(std::string(result.gl_pathv[i]));
  }
  globfree(&result);
  return ret;
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
