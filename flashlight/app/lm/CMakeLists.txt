cmake_minimum_required(VERSION 3.10)

# ------------------------- Library -------------------------
add_library(
  flashlight-app-lm
  ""
  )

# ------------------------ Global External Dependencies ------------------------
# Find gflags
find_package(GLOG REQUIRED)
if (GLOG_FOUND)
  message(STATUS "GLOG found")
else()
  message(FATAL_ERROR "GLOG not found")
endif()

find_package(gflags REQUIRED)
if (gflags_FOUND)
  message(STATUS "gflags found")
else()
  message(FATAL_ERROR "gflags not found")
endif()

target_sources(
  flashlight-app-lm
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Trainer.cpp
  )

target_link_libraries(
  flashlight-app-lm
  PUBLIC
  ${GLOG_LIBRARIES}
  ${gflags_LIBRARIES}
  flashlight
  fl-libraries
  )

target_include_directories(
  flashlight-app-lm
  PUBLIC
  ${GLOG_INCLUDE_DIRS}
  ${gflags_LIBRARIES}
  )
# ------------------------ Components ------------------------

# Common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# Data
include(${CMAKE_CURRENT_LIST_DIR}/data/CMakeLists.txt)

# ----------------------------- Binaries -----------------------------
add_executable(fl_lm_train ${CMAKE_CURRENT_LIST_DIR}/Train.cpp)
add_executable(
  fl_lm_dictionary_builder
  ${CMAKE_CURRENT_LIST_DIR}/BuildDictionary.cpp
  )

target_link_libraries(fl_lm_train flashlight-app-lm)
target_link_libraries(fl_lm_dictionary_builder flashlight-app-lm)

set_executable_output_directory(fl_lm_train "${FL_BUILD_BINARY_OUTPUT_DIR}/lm")
set_executable_output_directory(
  fl_lm_dictionary_builder
  "${FL_BUILD_BINARY_OUTPUT_DIR}/lm"
  )

install(TARGETS fl_lm_train RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(
  TARGETS
  fl_lm_dictionary_builder
  RUNTIME
  DESTINATION
  ${FL_INSTALL_BIN_DIR}
  )

# --------------------------- Tests ---------------------------
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
