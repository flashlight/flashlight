cmake_minimum_required(VERSION 3.10)

set(FL_PKG_VISION_DIR ${FL_PKG_DIR}/vision)
include(${FL_PKG_VISION_DIR}/CMakeLists.txt)

set(FL_PKG_RUNTIME_DIR ${FL_PKG_DIR}/runtime)
include(${FL_PKG_RUNTIME_DIR}/CMakeLists.txt)
