cmake_minimum_required(VERSION 3.10)

# ------------------------- Library -------------------------
add_library(
  flashlight-app-asr
  ""
  )

# ------------------------- Dependency Fallback -------------------------

if (FL_LIBRARIES_USE_MKL)
  find_package(MKL)
  if (NOT MKL_FOUND)
    message(WARNING "MKL not found; forcing FL_LIBRARIES_USE_MKL=OFF.")
    set(FL_LIBRARIES_USE_MKL OFF CACHE BOOL "" FORCE)
  endif ()
endif ()

# ------------------------ Global External Dependencies ------------------------
# Find GLog
find_package(GLOG REQUIRED)
if (GLOG_FOUND)
  message(STATUS "GLOG found")
else()
  message(FATAL_ERROR "GLOG not found")
endif()

# Find gflags
find_package(gflags REQUIRED)
if (gflags_FOUND)
  message(STATUS "gflags found")
else()
  message(FATAL_ERROR "gflags not found")
endif()

target_link_libraries(
  flashlight-app-asr
  PUBLIC
  ${GLOG_LIBRARIES}
  ${gflags_LIBRARIES}
  ${MKL_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-asr
  PUBLIC
  ${gflags_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${MKL_INCLUDE_DIR}
  )

# ------------------------ Components ------------------------

# Common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# Criterion
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

# Data
include(${CMAKE_CURRENT_LIST_DIR}/data/CMakeLists.txt)

# Decoder
include(${CMAKE_CURRENT_LIST_DIR}/decoder/CMakeLists.txt)

# Runtime
include(${CMAKE_CURRENT_LIST_DIR}/runtime/CMakeLists.txt)

# Augmentation
include(${CMAKE_CURRENT_LIST_DIR}/augmentation/CMakeLists.txt)

# ----------------------------- Binaries -----------------------------
add_executable(fl_asr_train ${CMAKE_CURRENT_LIST_DIR}/Train.cpp)
add_executable(fl_asr_test ${CMAKE_CURRENT_LIST_DIR}/Test.cpp)
add_executable(fl_asr_decode ${CMAKE_CURRENT_LIST_DIR}/Decode.cpp)

target_link_libraries(fl_asr_train flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(fl_asr_test flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(fl_asr_decode flashlight-app-asr ${CMAKE_DL_LIBS})

set_executable_output_directory(fl_asr_train "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_test "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_decode "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")

install(TARGETS fl_asr_train RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(TARGETS fl_asr_test RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(TARGETS fl_asr_decode RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})

# ----------------------------- Tutorial Binaries -----------------------------
add_executable(fl_asr_tutorial_inference_ctc ${CMAKE_CURRENT_LIST_DIR}/tutorial/InferenceCTC.cpp)
target_link_libraries(fl_asr_tutorial_inference_ctc flashlight-app-asr ${CMAKE_DL_LIBS})
set_executable_output_directory(fl_asr_tutorial_inference_ctc "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")

# --------------------------- Tests ---------------------------

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
