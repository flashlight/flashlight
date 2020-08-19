cmake_minimum_required(VERSION 3.5.1)

add_library(
  libraries-audio-criterion
  INTERFACE
  )

target_sources(
  libraries-audio-criterion
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/CriterionUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/ForceAlignmentCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/ConnectionistTemporalClassificationCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/FullConnectionCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/ViterbiPath.cpp
  )

target_link_libraries(
  libraries-audio-criterion
  INTERFACE
  libraries-common
  )

# ------------------------- CUDA-specific -------------------------
if (FL_LIBRARIES_USE_CUDA)
  include(${CMAKE_MODULE_PATH}/BuildCUB.cmake)
  include(${CMAKE_MODULE_PATH}/CUDAUtils.cmake)

  set_cuda_cxx_compile_flags()
  set_cuda_arch_nvcc_flags()

  # hacky: add -fPIC to nvcc flags if needed
  if (FL_BUILD_FOR_PYTHON OR CMAKE_POSITION_INDEPENDENT_CODE)
    cuda_enable_position_independent_code()
  endif ()

  cuda_include_directories(
    ${CUB_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    )

  # Qualify name because CUDA target is public
  cuda_add_library(
    libraries-audio-criterion-cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/CriterionUtils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ForceAlignmentCriterion.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/FullConnectionCriterion.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ViterbiPath.cu
    )

  add_dependencies(libraries-audio-criterion-cuda CUB)

  target_link_libraries(
    libraries-audio-criterion
    INTERFACE
    ${CUDA_LIBRARIES}
    libraries-audio-criterion-cuda
    )

  target_include_directories(
    libraries-audio-criterion
    INTERFACE
    ${CUDA_INCLUDE_DIRS}
    )
else ()
  message(
    FATAL_ERROR
    "wav2letter criterions not implemented for non-CUDA backends"
    )
endif ()
