cmake_minimum_required(VERSION 3.16)

option(FL_BUILD_DISTRIBUTED
  "Build distributed computation capabilities with Flashlight" ON)

option(FL_USE_NCCL "Build with NCCL for distributed computation" OFF)
option(FL_USE_GLOO "Build with Gloo for distributed computation" OFF)

# TODO: relax this
if (FL_USE_NCCL AND FL_USE_GLOO)
  message(STATUS "Cannot build multiple distributed backends simultaneously")
endif()

if (FL_BUILD_DISTRIBUTED)
  # If using CUDA, enable NCCL, else build with Gloo
  if (FL_USE_CUDA)
    set(FL_USE_NCCL ON)
  elseif (FL_USE_CPU)
    set(FL_USE_GLOO ON)
  else()
    # A stub impl that throws/executes noops for most distributed ops
    set(FL_DISTRIBUTED_STUB ON)
  endif()
endif ()

# Build sources only in distributed mode. Distributed headers will be included
# regardless but usage of the APIs will fail to link if not enabled.
# TODO: conditionally install headers?
if (FL_BUILD_DISTRIBUTED)
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/DistributedApi.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FileStore.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reducers/InlineReducer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reducers/CoalescingReducer.cpp
    )
endif()

if (FL_DISTRIBUTED_STUB)
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/stub/DistributedBackend.cpp
    )
endif ()

# ----------------------------- Dependencies -----------------------------
# MPI is required for any distributed support
# TODO: make this and the rendezvous impl configurable
if (FL_BUILD_DISTRIBUTED)
  find_package(MPI REQUIRED)
  target_link_libraries(flashlight PUBLIC MPI::MPI_CXX)
endif ()

if (FL_USE_GLOO)
  find_package(Gloo CONFIG)
  if (NOT Gloo_FOUND)
    if (NOT FL_BUILD_STANDALONE)
      message(FATAL_ERROR "Required dependency Gloo not found")
    endif()
    message(STATUS "Gloo not found - downloading and building from source")
    include(${PROJECT_SOURCE_DIR}/cmake/BuildGloo.cmake)
  endif()

  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/DistributedBackend.cpp
    )

  target_link_libraries(flashlight PRIVATE gloo)
endif()

if (FL_USE_NCCL)
  find_package(NCCL REQUIRED)
  if (NCCL_FOUND)
    message(STATUS "NCCL found: (include: ${NCCL_INCLUDE_DIRS} lib: ${NCCL_LIBRARIES}")
    setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindNCCL.cmake)
  endif()

  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/DistributedBackend.cpp
    )

  target_link_libraries(flashlight PRIVATE ${NCCL_LIBRARIES})
  target_include_directories(flashlight PRIVATE ${NCCL_INCLUDE_DIRS})

  target_compile_definitions(
    flashlight
    PUBLIC
    "-DNO_NCCL_COMM_DESTROY_HANDLE"
    )
endif()
