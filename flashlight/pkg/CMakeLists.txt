cmake_minimum_required(VERSION 3.10)

fl_dependent_option(
  FL_BUILD_ALL_PKGS
  "Build all flashlight packages"
  ON
  "FL_BUILD_CORE;FL_BUILD_CONTRIB"
  OFF
)

# --------------------------- Runtime  ---------------------------
fl_dependent_option(
  FL_BUILD_PKG_RUNTIME
  "Build runtime package for Flashlight"
  "${FL_BUILD_ALL_PKGS}"
  "FL_BUILD_CORE;FL_BUILD_CONTRIB"
  OFF)
if(FL_BUILD_PKG_RUNTIME)
  set(FL_PKG_RUNTIME_DIR ${FL_PKG_DIR}/runtime)
  include(${FL_PKG_RUNTIME_DIR}/CMakeLists.txt)
  set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_pkg_runtime)
  setup_install_headers(${FL_PKG_RUNTIME_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)
endif()

# --------------------------- Vision  ---------------------------
fl_dependent_option(
  FL_BUILD_PKG_VISION
  "Build vision package for Flashlight"
  "${FL_BUILD_ALL_PKGS}"
  "FL_BUILD_CORE;FL_BUILD_CONTRIB;FL_BUILD_LIB_COMMON;FL_BUILD_LIB_SET"
  OFF)
if(FL_BUILD_PKG_VISION)
  set(FL_PKG_VISION_DIR ${FL_PKG_DIR}/vision)
  include(${FL_PKG_VISION_DIR}/CMakeLists.txt)
  set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_pkg_vision)
  setup_install_headers(${FL_PKG_VISION_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)
endif()

# --------------------------- Text  ---------------------------
fl_dependent_option(
  FL_BUILD_PKG_TEXT
  "Build text package for Flashlight"
  "${FL_BUILD_ALL_PKGS}"
  "FL_BUILD_CORE;FL_BUILD_CONTRIB"
  OFF)
if(FL_BUILD_PKG_TEXT)
  set(FL_PKG_TEXT_DIR ${FL_PKG_DIR}/text)
  include(${FL_PKG_TEXT_DIR}/CMakeLists.txt)
  set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_pkg_text)
  setup_install_headers(${FL_PKG_TEXT_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)
endif()

# --------------------------- Speech  ---------------------------
fl_dependent_option(
  FL_BUILD_PKG_SPEECH
  "Build speech package for Flashlight"
  "${FL_BUILD_ALL_PKGS}"
  "FL_BUILD_CORE;FL_BUILD_CONTRIB;FL_BUILD_LIB_AUDIO"
  OFF)

if(FL_BUILD_PKG_SPEECH)
  set(FL_PKG_SPEECH_DIR ${FL_PKG_DIR}/speech)
  include(${FL_PKG_SPEECH_DIR}/CMakeLists.txt)
  set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_pkg_speech)
  setup_install_headers(${FL_PKG_SPEECH_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)
endif()

# --------------------------- Halide  ---------------------------
fl_dependent_option(
  FL_BUILD_PKG_HALIDE
  "Build speech package for Flashlight"
  OFF # for right now, don't build halide pkg with all pkgs to make CI easier
  "FL_BUILD_CORE;FL_BUILD_CONTRIB"
  OFF)
if(FL_BUILD_PKG_HALIDE)
  set(FL_PKG_HALIDE_DIR ${FL_PKG_DIR}/halide)
  include(${FL_PKG_HALIDE_DIR}/CMakeLists.txt)
  set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_pkg_halide)
  setup_install_headers(${FL_PKG_HALIDE_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)
endif()
