/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace fl {
namespace detail {

/**
 * An RAII abstraction to start and stop profiling recording.
 */
class ScopedProfiler {
 public:
  ScopedProfiler();
  ~ScopedProfiler();
};

/**
 * An RAII abstractiont to label a profile interval over the lifetime for an
 object given a specific scope. For example:
 * \code
   {
     ProfileTracer tr("myOperation");
     // some code which does stuff and will be labeled
     // as myOperation
   }
 * \endcode
 */
class ProfileTracer {
 public:
  ProfileTracer(const std::string& name);
  ~ProfileTracer();
};

} // namespace detail
} // namespace fl

#if FL_BUILD_PROFILING
// Used to generate a unique name for the expansion
#define _FL_PROFILE_CAT(a, b) a##b

#define FL_PROFILE_TRACE(name) \
  fl::detail::ProfileTracer _FL_PROFILE_CAT(profileTracer, __LINE__)(name);

#define FL_SCOPED_PROFILE() \
  fl::detail::ScopedProfiler _FL_PROFILE_CAT(scopedProfile, __LINE__);

#else
#define FL_PROFILE_TRACE(_)
#define FL_SCOPED_PROFILE()
#endif
