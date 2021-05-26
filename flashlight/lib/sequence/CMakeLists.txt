cmake_minimum_required(VERSION 3.10)

# ------------------------- Components -------------------------

add_library(
  fl_lib_sequence
  ""
  )

target_include_directories(
  fl_lib_sequence
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  )

# Only build the library if the FL CUDA backend is enabled.
if (NOT FL_USE_CUDA)
  message(STATUS "FL_BACKEND is not CUDA - forcing FL_LIBRARIES_USE_CUDA to OFF")
  set(FL_LIBRARIES_USE_CUDA OFF)
endif()

# criterion-library
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

set(FL_LIBRARIES_SEQUENCE_CUDA_SOURCES
  ${FL_LIBRARIES_SEQUENCE_CRITERION_CUDA_SOURCES}
  )
set(FL_LIBRARY_CUDA_SOURCES
  ${FL_LIBRARIES_SEQUENCE_CUDA_SOURCES}
  )

target_include_directories(
  fl_lib_sequence
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  )

if (FL_LIBRARIES_USE_CUDA)
message("here!!! using")
  target_sources(
    fl_lib_sequence
    PRIVATE
    ${FL_LIBRARY_CUDA_SOURCES}
    )

  # CUB
  if(CUDA_VERSION_MAJOR VERSION_LESS 11)
    find_package(cub)
    if (NOT cub_FOUND)
      message(STATUS "Could not find cub - will download from source")
      include(${CMAKE_MODULE_PATH}/BuildCUB.cmake)
      add_dependencies(fl-libraries CUB)
    else ()
      message(STATUS "Found cub: (include: ${cub_INCLUDE_DIRS})")
    endif()
  endif()

  target_include_directories(
    fl_lib_sequence
    PUBLIC
    $<BUILD_INTERFACE:${cub_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )

  target_link_libraries(
    fl_lib_sequence
    PUBLIC
    ${CUDA_LIBRARIES}
    )
endif()

# --------------------------- Tests ---------------------------

#if (FL_BUILD_TESTS)
  #add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
#endif ()


