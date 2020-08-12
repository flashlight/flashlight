cmake_minimum_required(VERSION 3.5.1)

option(FL_BUILD_APP_ASR "Build asr task for flashlight" ON)

if (FL_BUILD_APP_ASR)
  if (NOT FL_BUILD_DISTRIBUTED)
    message(FATAL_ERROR "FL_BUILD_DISTRIBUTED must be enabled to build speech recognition")
  endif()
  set(FL_APP_ASR_ROOT_DIR ${FLASHLIGHT_APPS_DIR}/asr)
  include(${FL_APP_ASR_ROOT_DIR}/CMakeLists.txt)
endif ()
