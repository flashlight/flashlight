cmake_minimum_required(VERSION 3.10)

set(
  DISTRIBUTED_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/DistributedApi.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FileStore.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reducers/InlineReducer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reducers/CoalescingReducer.cpp
  )

# Build sources only in distributed mode. Distributed headers will be included regardless,
# but usage of the apis will fail to link if not enabled.
if (FL_BUILD_DISTRIBUTED)
  target_sources(
    flashlight
    PRIVATE
    ${DISTRIBUTED_SOURCES}
    )
endif()

# ----------------------------- Dependencies -----------------------------
# Gloo
if (USE_GLOO)
  find_package(Gloo CONFIG)
  if (NOT Gloo_FOUND AND FL_BUILD_STANDALONE)
    message(STATUS "Gloo not found - downloading and building from source")
    include(${CMAKE_MODULE_PATH}/BuildGloo.cmake)
    add_dependencies(flashlight Gloo) # ExternalProject target
  endif()
endif()

# NCCL
if (USE_NCCL)
  find_package(NCCL REQUIRED)
  if (NCCL_FOUND)
    message(STATUS "NCCL found: (include: ${NCCL_INCLUDE_DIRS} lib: ${NCCL_LIBRARIES}")
    setup_install_find_module(${CMAKE_MODULE_PATH}/FindNCCL.cmake)
  endif()
endif()

# RCCL
if (USE_RCCL)
  find_package(RCCL REQUIRED)
  if (RCCL_FOUND)
    message(STATUS "RCCL found: (include: ${RCCL_INCLUDE_DIRS} lib: ${RCCL_LIBRARIES}")
    setup_install_find_module(${CMAKE_MODULE_PATH}/FindRCCL.cmake)
  endif()
endif()

# MPI
find_package(MPI)
if (MPI_C_FOUND AND MPI_CXX_FOUND)
  message(STATUS "MPI_VERSION found: ${MPI_VERSION} ${MPI_C_VERSION_MAJOR}.${MPI_C_VERSION_MINOR}")
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

  set(MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH} ${MPI_INCLUDE_PATH})
else()
  message(STATUS "MPI not found")
  if (FL_BUILD_DISTRIBUTED)
    message(FATAL_ERROR "MPI_C and MPI_CXX not found; required to build flashlight distributed")
  endif()
endif()

# ----------------------------- Backend libs -----------------------------
if (FL_BUILD_DISTRIBUTED)
  target_link_libraries(
    flashlight
    PUBLIC
    ${MPI_LIBRARIES}
    )
message("-------------------- MPI_INCLUDE_DIRS=${MPI_INCLUDE_DIRS}")
message("-------------------- MPI_LIBRARIES=${MPI_LIBRARIES}")
  target_include_directories(
    flashlight
    PUBLIC
    ${MPI_INCLUDE_DIRS}
    )
endif ()

# Distributed
if (USE_NCCL)
  set(
    DISTRIBUTED_NCCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/DistributedBackend.cpp
    )

  target_sources(
    flashlight
    PRIVATE
    ${DISTRIBUTED_NCCL_SOURCES}
    )

  target_link_libraries(
    flashlight
    PRIVATE
    ${CUDA_LIBRARIES}
    ${NCCL_LIBRARIES}
    )

  target_include_directories(
    flashlight
    PRIVATE
    ${NCCL_INCLUDE_DIRS}
    )

  target_compile_definitions(
    flashlight
    PUBLIC
    "-DNO_NCCL_COMM_DESTROY_HANDLE"
    )
endif ()

# Distributed
if (USE_RCCL)
  set(
    DISTRIBUTED_RCCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/miopen/DistributedBackend.cpp
    )

  target_sources(
    flashlight
    PRIVATE
    ${DISTRIBUTED_RCCL_SOURCES}
    )

  target_link_libraries(
    flashlight
    PRIVATE
    ${MIOPEN_LIBRARIES}
    ${RCCL_LIBRARIES}
    )

  target_include_directories(
    flashlight
    PRIVATE
    ${RCCL_INCLUDE_DIRS}
    )

  target_compile_definitions(
    flashlight
    PUBLIC
    "-DNO_RCCL_COMM_DESTROY_HANDLE"
    )
endif ()

# Distributed
if (USE_GLOO)
  set(
    DISTRIBUTED_GLOO_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/DistributedBackend.cpp
    )

  target_sources(
    flashlight
    PRIVATE
    ${DISTRIBUTED_GLOO_SOURCES}
    )

  target_link_libraries(
    flashlight
    PRIVATE
    gloo
    )
endif ()
