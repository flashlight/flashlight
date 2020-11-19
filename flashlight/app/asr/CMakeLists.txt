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
  ${gflags_LIBRARIES}
  ${MKL_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-asr
  PUBLIC
  ${gflags_INCLUDE_DIRS}
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

# ----------------------------- Binaries -----------------------------
add_executable(asr_train ${CMAKE_CURRENT_LIST_DIR}/Train.cpp)
add_executable(asr_test ${CMAKE_CURRENT_LIST_DIR}/Test.cpp)
add_executable(asr_decode ${CMAKE_CURRENT_LIST_DIR}/Decode.cpp)

target_link_libraries(asr_train flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(asr_test flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(asr_decode flashlight-app-asr ${CMAKE_DL_LIBS})

# --------------------------- Tests ---------------------------

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
