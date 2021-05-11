cmake_minimum_required(VERSION 3.10)

set(FL_TENSOR_BACKEND "AF" CACHE STRING "Tensor backend with which to build flashlight")
# Select from exactly one tensor backend
set_property(CACHE FL_TENSOR_BACKEND PROPERTY STRINGS AF)
# Map to flags
set(FL_USE_ARRAYFIRE OFF)
if (FL_TENSOR_BACKEND STREQUAL "AF")
  set(FL_USE_ARRAYFIRE ON)
else()
  message(FATAL_ERROR "Invalid FL_TENSOR_BACKEND specified")
endif()

if (FL_USE_ARRAYFIRE)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/af/CMakeLists.txt)
endif()

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Types.cpp
)
