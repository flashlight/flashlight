cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Components -------------------------

# decoder
include(${CMAKE_CURRENT_LIST_DIR}/decoder/CMakeLists.txt)

# dictionary
include(${CMAKE_CURRENT_LIST_DIR}/dictionary/CMakeLists.txt)

# tokenizer
include(${CMAKE_CURRENT_LIST_DIR}/tokenizer/CMakeLists.txt)
 
