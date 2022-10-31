cmake_minimum_required(VERSION 3.16)

# --------------------------- Lib Options  ---------------------------
# Option to allow users to essentially be opt-opt. You can still selectively
# turn of libraries with something like
# `FL_BUILD_ALL_LIBS=ON -DFL_BUILD_LIB_AUDIO=OFF` to build everything but the
# audio library
option(FL_BUILD_ALL_LIBS "Build flashlight libraries" OFF)

# audio
option(
  FL_BUILD_LIB_AUDIO
  "Build flashlight audio library"
  ${FL_BUILD_ALL_LIBS})
if(FL_BUILD_LIB_AUDIO)
  set(FL_AUDIO_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/audio)
  include(${FL_AUDIO_LIB_DIR}/CMakeLists.txt)
  list(APPEND INSTALLABLE_TARGETS fl_lib_audio)
  setup_install_headers(${FL_AUDIO_LIB_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC})
endif()
