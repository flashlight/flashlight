/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/*
 * This header shim for std::filesystem exists to support compatibility with gcc
 * >= 7 and gcc < 8 (and most other compilers with old distributions that are
 * still in use).
 *
 * Will be removed when gcc 9 is a supported minimum in Flashlight.
 */
#if defined(__cpp_lib_filesystem)
#define _USE_STD_FS_EXP 0
#elif defined(__cpp_lib_experimental_filesystem)
#define _USE_STD_FS_EXP 1
#elif !defined(__has_include)
#define _USE_STD_FS_EXP 1 // just use experimental
#elif __has_include(<filesystem>)
#define _USE_STD_FS_EXP 0
#elif __has_include(<experimental/filesystem>)
#define _USE_STD_FS_EXP 1
#else
#error Could not find either system header "<filesystem>" or "<experimental/filesystem>"
#endif

// Include
#if _USE_STD_FS_EXP
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#undef _USE_STD_FS_EXP
