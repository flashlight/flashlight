/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
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

std::ifstream createInputStream(const std::string& filename);

std::ofstream createOutputStream(
    const std::string& filename,
    std::ios_base::openmode mode = std::ios_base::out);
} // namespace lib
} // namespace fl
