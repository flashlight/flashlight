cmake_minimum_required(VERSION 3.10)

set(FL_PKG_VISION_DIR ${FL_PKG_DIR}/vision)
include(${FL_PKG_VISION_DIR}/CMakeLists.txt)

set(FL_PKG_RUNTIME_DIR ${FL_PKG_DIR}/runtime)
include(${FL_PKG_RUNTIME_DIR}/CMakeLists.txt)

set(FL_PKG_SPEECH_DIR ${FL_PKG_DIR}/speech)
include(${FL_PKG_SPEECH_DIR}/CMakeLists.txt)

set(FL_PKG_TEXT_DIR ${FL_PKG_DIR}/text)
include(${FL_PKG_TEXT_DIR}/CMakeLists.txt)

if(FL_BUILD_PKG_HALIDE)
  set(FL_PKG_HALIDE_DIR ${FL_PKG_DIR}/halide)
  include(${FL_PKG_HALIDE_DIR}/CMakeLists.txt)
endif()
