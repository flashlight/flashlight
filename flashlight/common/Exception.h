/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/exception.h>
#include <glog/logging.h>

#define AFML_THROW_ERR(__msg, __err)       \
  do {                                     \
    LOG(FATAL) << __err << " : " << __msg; \
  } while (0)

#define AFML_ASSERT(__chk, __msg, __err) \
  if (!(__chk)) {                        \
    AFML_THROW_ERR(__msg, __err);        \
  }
