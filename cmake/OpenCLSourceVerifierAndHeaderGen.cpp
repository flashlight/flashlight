/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <af/opencl.h>

void usage(const std::string& exeName) {
  std::cerr
      << "usage:\n"
      << exeName << "\n\t--input=[opencl source file] "
      << "\n\t--output=[generated header file name] "
      << "\n\t--var=[name of the char* variable that points to the source]"
      << "\n\t--namespace=[namespace for the variable]" << std::endl;
}

#define FL_OPENCL_CHECK(err, inputFile)            \
  {                                                \
    if (err != CL_SUCCESS) {                       \
      std::stringstream ss;                        \
      ss << "Source file=" << inputFile            \
         << " has OpenCL compile error=" << (err); \
      throw std::runtime_error(ss.str());          \
    }                                              \
  }

/**
 * Calls the driver to compile the opencl kernel source.
 * Throws an exception whe source files to compile.
 */
void verifySourceIsCompiling(
    const std::string& source,
    const std::string& inputKernelFile) {
  cl_context context_ = nullptr;
  cl_device_id deviceId_ = nullptr;
  cl_command_queue queue_ = nullptr;
  FL_OPENCL_CHECK(
      afcl_get_context(&context_, /*retain=*/false), inputKernelFile);
  FL_OPENCL_CHECK(afcl_get_device_id(&deviceId_), inputKernelFile);
  FL_OPENCL_CHECK(afcl_get_queue(&queue_, /*retain=*/false), inputKernelFile);

  const char* pSource = source.c_str();
  cl_int status = CL_SUCCESS;
  cl_program program =
      clCreateProgramWithSource(context_, 1, &pSource, nullptr, &status);
  FL_OPENCL_CHECK(status, inputKernelFile);
  std::cerr << inputKernelFile << " compile status=" <<
      clBuildProgram(program, 1, &deviceId_, nullptr, nullptr, nullptr);
}

std::string readFileIntoString(const std::string& inputKernelFile) {
  std::ifstream input(inputKernelFile);
  if (!input) {
    std::stringstream ss;
    ss << "failed to open input kernel file=" << inputKernelFile;
    throw std::runtime_error(ss.str());
  }
  std::string str;

  input.seekg(0, std::ios::end);
  str.reserve(input.tellg());
  input.seekg(0, std::ios::beg);

  str.assign(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  return str;
}

/**
 * tokenize string on delimiter.
 */
std::vector<std::string> tokenize(
    const std::string& str,
    const std::string& delim) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;
  do {
    pos = str.find(delim, prev);
    if (pos == std::string::npos) {
      pos = str.length();
    }
    std::string token = str.substr(prev, pos - prev);
    if (!token.empty()) {
      tokens.push_back(token);
    }
    prev = pos + delim.length();
  } while (pos < str.length() && prev < str.length());
  return tokens;
}

std::map<std::string, std::string> parseFlags(int argc, char* argv[]) {
  std::map<std::string, std::string> result;
  for (int i = 1; i < argc; ++i) {
    std::cout << argv[i] << " ";
    const std::vector<std::string> tokens = tokenize(argv[i], "=");
    if (tokens.size() == 2 && tokens[0].find("--") == 0) {
      result[tokens[0]] = tokens[1];
    } else {
      std::stringstream ss;
      ss << "invalid flag format=" << argv[i];
      throw std::invalid_argument(ss.str());
    }
  }
  std::cout << std::endl;
  return result;
}

std::string getFlagValue(
    const std::map<std::string, std::string>& flagToValue,
    const std::string& flag,
    const std::string& exeName) {
  auto itr = flagToValue.find(flag);
  if (itr == flagToValue.end()) {
    usage(exeName);
    std::stringstream ss;
    ss << "missing flag=" << flag;
    throw std::invalid_argument(ss.str());
  }
  return itr->second;
}

void writeHeaderFile(
    const std::string& kernelSourceCode,
    const std::string& ouputKernelHeaderFile,
    const std::string& namespaceName,
    const std::string& varName) {
  std::ofstream output(ouputKernelHeaderFile);
  if (!output) {
    std::stringstream ss;
    ss << "failed to open output kernel file=" << ouputKernelHeaderFile;
    throw std::runtime_error(ss.str());
  }

  output << "#pragma once\n\n"
         << "namespace " << namespaceName << "{\n\n"
         << "const char* const " << varName << " =R\"rawStringKernel("
         << std::endl
         << kernelSourceCode << "\n)rawStringKernel\";\n\n"
         << "} // namespace " << namespaceName << std::endl;
}

int main(int argc, char* argv[]) {
  const auto flagToValue = parseFlags(argc, argv);
  const auto inputKernelFile = getFlagValue(flagToValue, "--input", argv[0]);
  const auto ouputKernelHeaderFile =
      getFlagValue(flagToValue, "--output", argv[0]);
  const auto namespaceName = getFlagValue(flagToValue, "--namespace", argv[0]);
  const auto varName = getFlagValue(flagToValue, "--var", argv[0]);

  const auto kernelSourceCode = readFileIntoString(inputKernelFile);
  // verifySourceIsCompiling(kernelSourceCode, inputKernelFile);
  writeHeaderFile(
      kernelSourceCode, ouputKernelHeaderFile, namespaceName, varName);
}
