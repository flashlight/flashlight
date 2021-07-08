cmake_minimum_required(VERSION 3.10)

fl_dependent_option(
  FL_BUILD_ALL_APPS
  "Set all Flashlight apps ON by default"
  OFF
  "FL_BUILD_CORE;FL_BUILD_CONTRIB;"
  OFF)

# --------------------------- ASR  ---------------------------
fl_dependent_option(
  FL_BUILD_APP_ASR
  "Build asr task for Flashlight"
  ${FL_BUILD_ALL_APPS}
  "FL_BUILD_CORE;FL_BUILD_CONTRIB;FL_BUILD_PKG_VISION;FL_BUILD_LIB_COMMON;FL_BUILD_LIB_SET"
  OFF)
if (FL_BUILD_APP_ASR)
  message(STATUS "Building flashlight ASR app")
  if (NOT FL_BUILD_DISTRIBUTED)
    message(FATAL_ERROR "FL_BUILD_DISTRIBUTED must be enabled to build speech recognition")
  endif()
  set(FL_APP_ASR_ROOT_DIR ${FL_APPS_DIR}/asr)
  include(${FL_APP_ASR_ROOT_DIR}/CMakeLists.txt)
endif ()

# --------------------------- Image Classificiation  ---------------------------
fl_dependent_option(
  FL_BUILD_APP_IMGCLASS
  "Build image classification app"
  "${FL_BUILD_ALL_APPS}"
  "FL_BUILD_CORE;FL_BUILD_CONTRIB;FL_BUILD_PKG_VISION;FL_BUILD_PKG_RUNTIME"
  OFF)
if (FL_BUILD_APP_IMGCLASS) # imgclass is required to build objdet
  message(STATUS "Building flashlight Image Classification app")
  if (NOT FL_BUILD_DISTRIBUTED)
    message(FATAL_ERROR "FL_BUILD_DISTRIBUTED must be enabled for image classification")
  endif()
  set(FL_APP_IMGCLASS_ROOT_DIR ${FL_APPS_DIR}/imgclass)
  include(${FL_APP_IMGCLASS_ROOT_DIR}/CMakeLists.txt)
endif ()

# --------------------------- LM  ---------------------------
fl_dependent_option(
  FL_BUILD_APP_LM
  "Build image classification app"
  "${FL_BUILD_ALL_APPS}"
  "FL_BUILD_CORE;FL_BUILD_CONTRIB;FL_BUILD_PKG_TEXT;FL_BUILD_PKG_VISION;FL_BUILD_PKG_RUNTIME"
  OFF)
if (FL_BUILD_APP_LM)
  message(STATUS "Building flashlight LM app")
  if (NOT FL_BUILD_DISTRIBUTED)
    message(FATAL_ERROR "FL_BUILD_DISTRIBUTED must be enabled to language modeling")
  endif()
  set(FL_APP_LM_ROOT_DIR ${FL_APPS_DIR}/lm)
  include(${FL_APP_LM_ROOT_DIR}/CMakeLists.txt)
endif ()

# --------------------------- Objdet Detection  ---------------------------
fl_dependent_option(
  FL_BUILD_APP_OBJDET
  "Build image classification app"
  "${FL_BUILD_ALL_APPS}"
  "FL_BUILD_CORE;FL_BUILD_CONTRIB;FL_BUILD_PKG_VISION;FL_BUILD_PKG_RUNTIME"
  OFF)
if (FL_BUILD_APP_OBJDET)
  message(STATUS "Building flashlight Object Detection app")
  if (NOT FL_BUILD_DISTRIBUTED)
    message(FATAL_ERROR
    "FL_BUILD_DISTRIBUTED must be enabled for object detection")
  endif()
  set(FL_APP_OBJDET_ROOT_DIR ${FL_APPS_DIR}/objdet)
  include(${FL_APP_OBJDET_ROOT_DIR}/CMakeLists.txt)
endif ()

# Build benchmark additionally
if (FL_BUILD_ALL_APPS)
  message(STATUS "Building flashlight app benchmark")
  if (NOT FL_BUILD_DISTRIBUTED)
    message(FATAL_ERROR
    "FL_BUILD_DISTRIBUTED must be enabled for benchmarking")
  endif()
  set(FL_APP_BENCHMARK_ROOT_DIR ${FL_APPS_DIR}/benchmark)
  include(${FL_APP_BENCHMARK_ROOT_DIR}/CMakeLists.txt)
  set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} flashlight-app-benchmark)
  setup_install_headers(${FL_APP_BENCHMARK_ROOT_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/app)
endif()
