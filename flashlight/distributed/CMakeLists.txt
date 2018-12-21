cmake_minimum_required(VERSION 3.5.1)

add_library(
  Distributed
  INTERFACE
  )

set(
  DISTRIBUTED_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/distributed/DistributedApi.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/distributed/backend/utils/FileStore.cpp
  )

# Build sources only in distributed mode. Distributed headers will be included regardless,
# but usage of the apis will fail to link if not enabled.
if (FL_BUILD_DISTRIBUTED)
  target_sources(
    Distributed
    INTERFACE
    $<BUILD_INTERFACE:${DISTRIBUTED_SOURCES}>
    )
endif()

# ----------------------------- Dependencies -----------------------------
# Gloo
find_package(Gloo QUIET)
if (Gloo_FOUND)
  message(STATUS "Gloo found")
else()
  message(STATUS "Gloo not found")
  if (USE_GLOO)
    message(FATAL_ERROR "Cannot build Gloo backend without Gloo")
  endif ()
endif()

# NCCL
find_package(NCCL QUIET)
if (NCCL_FOUND)
  message(STATUS "NCCL found")
else()
  message(STATUS "NCCL not found")
  if (USE_NCCL)
    message(FATAL_ERROR "Cannot build NCCL backend without NCCL")
  endif ()
endif()

# MPI
find_package(MPI)

if (MPI_C_FOUND AND MPI_CXX_FOUND)
  message(STATUS "MPI_CXX found")
  message(STATUS "MPI_CXX compile flags: " ${MPI_CXX_COMPILE_FLAGS})
  message(STATUS "MPI_CXX include path: " ${MPI_CXX_INCLUDE_PATH})
  message(STATUS "MPI_CXX LINK flags path: " ${MPI_CXX_LINK_FLAGS})
  message(STATUS "MPI_CXX libraries: " ${MPI_CXX_LIBRARIES})

  message(STATUS "MPI_C found")
  message(STATUS "MPI_C compile flags: " ${MPI_C_COMPILE_FLAGS})
  message(STATUS "MPI_C include path: " ${MPI_C_INCLUDE_PATH})
  message(STATUS "MPI_C LINK flags path: " ${MPI_C_LINK_FLAGS})
  message(STATUS "MPI_C libraries: " ${MPI_C_LIBRARIES})
else()
  message(FATAL_ERROR "MPI_CXX not found")
endif()


# ----------------------------- Backend libs -----------------------------
# Distributed
if (USE_NCCL)
  set(
    DISTRIBUTED_NCCL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/distributed/backend/cuda/DistributedBackend.cpp
    )
  
  target_sources(
    Distributed
    INTERFACE
    $<BUILD_INTERFACE:${DISTRIBUTED_NCCL_SOURCES}>  
    )

  target_link_libraries(
    Distributed
    INTERFACE
    ${CUDA_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${MPI_C_LIBRARIES}
    )

  target_include_directories(
    Distributed
    INTERFACE
    ${MPI_CXX_INCLUDE_PATH}
    ${NCCL_INCLUDE_DIRS}
    )
endif ()

# Distributed
if (USE_GLOO)
  set(
    DISTRIBUTED_GLOO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/distributed/backend/cpu/DistributedBackend.cpp
    )
  target_sources(
    Distributed
    INTERFACE
    $<BUILD_INTERFACE:${DISTRIBUTED_GLOO_SOURCES}>
    )

  target_link_libraries(
    Distributed
    INTERFACE
    ${MPI_CXX_LIBRARIES}
    ${MPI_C_LIBRARIES}
    ${Gloo_LIBRARY}
    )
 
  target_include_directories(
    Distributed
    INTERFACE
    ${MPI_CXX_INCLUDE_PATH}
    ${Gloo_INCLUDE_DIR}
    )
endif ()
