cmake_minimum_required(VERSION 3.10)

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
