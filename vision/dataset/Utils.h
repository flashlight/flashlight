#pragma once

#include "flashlight/dataset/datasets.h"
#include <glob.h>

namespace fl {
namespace cv {
namespace dataset {

/*
 * Small utility to glob for filepaths
 * @param[in] pat Filepattern to search for file names under
 */
inline std::vector<std::string> glob(const std::string& pat) {
  glob_t result;
  glob(pat.c_str(), GLOB_TILDE, nullptr, &result);
  std::vector<std::string> ret;
  for (unsigned int i = 0; i < result.gl_pathc; ++i) {
    ret.push_back(std::string(result.gl_pathv[i]));
  }
  globfree(&result);
  return ret;
}

} // namespace dataset
} // namespace cv
} // namespace fl
