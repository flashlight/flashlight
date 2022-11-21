cmake_minimum_required(VERSION 3.16)

# ----------------------------- OneDNN -----------------------------

# Try to find oneDNN
find_package(DNNL 2.5.2 CONFIG)
if (NOT DNNL_FOUND AND FL_USE_ONEDNN)
  message(FATAL_ERROR
    "oneDNN required to build oneDNN backend: oneDNN was not found")
else()
  message(STATUS "oneDNN found")
endif()

# TODO: remove this once the MKL dependency has been removed
# oneDNN backend requires MKL
find_package(MKL REQUIRED)
setup_install_find_module(${CMAKE_MODULE_PATH}/FindMKL.cmake)
set(FL_USE_MKL ON)

# ----------------------------- Sources -----------------------------

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnCPUStream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnTensor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)

target_include_directories(
  flashlight
  PUBLIC
  ${MKL_INCLUDE_DIR}
)

target_link_libraries(
  flashlight
  PUBLIC
  DNNL::dnnl
  ${MKL_LIBRARIES}
)
