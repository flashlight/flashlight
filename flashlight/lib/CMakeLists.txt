cmake_minimum_required(VERSION 3.10)

# ------------------------- Library -------------------------

#set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl-libraries)
set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_lib_audio)
set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_lib_common)
set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl_lib_sequence)

#target_compile_definitions(
  #fl-libraries
  #PUBLIC
  #${FL_COMPILE_DEFINITIONS}
  #)

# --------------------------- Components ---------------------------
# set
include(${CMAKE_CURRENT_LIST_DIR}/set/CMakeLists.txt)

# sequence
include(${CMAKE_CURRENT_LIST_DIR}/sequence/CMakeLists.txt)

# audio
include(${CMAKE_CURRENT_LIST_DIR}/audio/CMakeLists.txt)

# common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# text
include(${CMAKE_CURRENT_LIST_DIR}/text/CMakeLists.txt)

# ---------------------- CUDA Library -----------------------


# Accumulate CUDA sources after including library components because
# target_sources doesn't work with CUDA libraries as expected with
# old versions of CMake
# --------------------------- Tests ---------------------------

