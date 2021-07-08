cmake_minimum_required(VERSION 3.10)

# --------------------------- Lib Options  ---------------------------
# Build all libs by default. In order to selectively build libs,
# disable this option and opt-in to each using the individual build
# options. For example, if you are only building core, you only need
# the common library, so you can use -DFL_BUILD_ALL_LIBS=OFF and
# -DFL_BUILD_LIB_COMMON_ON
option(FL_BUILD_ALL_LIBS "Build flashlight libraries" OFF)

# --------------------------- Components ---------------------------
# set
option(FL_BUILD_LIB_SET "Build flashlight set library" ${FL_BUILD_ALL_LIBS})
if(FL_BUILD_LIB_SET)
  include(${CMAKE_CURRENT_LIST_DIR}/set/CMakeLists.txt)
endif()

# sequence
option(FL_BUILD_LIB_SEQUENCE "Build flashlight sequence library" ${FL_BUILD_ALL_LIBS})
if(FL_BUILD_LIB_SEQUENCE)
  include(${CMAKE_CURRENT_LIST_DIR}/sequence/CMakeLists.txt)
endif()

# audio
option(FL_BUILD_LIB_AUDIO "Build flashlight audio library" ${FL_BUILD_ALL_LIBS})
if(FL_BUILD_LIB_AUDIO)
  include(${CMAKE_CURRENT_LIST_DIR}/audio/CMakeLists.txt)
endif()

# common
option(FL_BUILD_LIB_COMMON "Build flashlight common library" ${FL_BUILD_ALL_LIBS})
if(FL_BUILD_LIB_COMMON)
  include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)
endif()

# text
option(FL_BUILD_LIB_TEXT "Build flashlight text library" ${FL_BUILD_ALL_LIBS})
if(FL_BUILD_LIB_TEXT)
  include(${CMAKE_CURRENT_LIST_DIR}/text/CMakeLists.txt)
endif()
