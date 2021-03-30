cmake_minimum_required(VERSION 3.10)

# ----------------------------- Dependencies -----------------------------
find_package(MKL REQUIRED)

# ----------------------------- Lib -----------------------------

target_sources(
  fl-libraries
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Functions.cpp
  )

target_link_libraries(
  fl-libraries
  PRIVATE
  ${MKL_LIBRARIES}
  )

target_include_directories(
  fl-libraries
  PRIVATE
  ${MKL_INCLUDE_DIR}
  )
