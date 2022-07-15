cmake_minimum_required(VERSION 3.10)

# ----------------------------- OneDNN -----------------------------

# Try to find oneDNN
find_package(DNNL 2.5.2 CONFIG)
if (NOT DNNL_FOUND AND FL_USE_ONEDNN)
  message(FATAL_ERROR
    "oneDNN required to build oneDNN backend: oneDNN was not found")
else()
  message(STATUS "oneDNN found")
endif()

# ----------------------------- Sources -----------------------------

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnTensor.cpp
)

target_link_libraries(
  flashlight
  PRIVATE
  DNNL::dnnl
)
