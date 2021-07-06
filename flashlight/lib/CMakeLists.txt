cmake_minimum_required(VERSION 3.10)

# --------------------------- Components ---------------------------
# set
if(FL_BUILD_LIB_SET)
  include(${CMAKE_CURRENT_LIST_DIR}/set/CMakeLists.txt)
endif()

# sequence
if(FL_BUILD_LIB_SEQUENCE)
  include(${CMAKE_CURRENT_LIST_DIR}/sequence/CMakeLists.txt)
endif()

# audio
if(FL_BUILD_LIB_AUDIO)
  include(${CMAKE_CURRENT_LIST_DIR}/audio/CMakeLists.txt)
endif()

# common
if(FL_BUILD_LIB_COMMON)
  include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)
endif()

# text
if(FL_BUILD_LIB_TEXT)
  include(${CMAKE_CURRENT_LIST_DIR}/text/CMakeLists.txt)
endif()
