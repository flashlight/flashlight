cmake_minimum_required(VERSION 3.5.1)

target_sources(
  fl-libraries
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/cpu/CriterionUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/ForceAlignmentCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/ConnectionistTemporalClassificationCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/FullConnectionCriterion.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/ViterbiPath.cpp
  )  

# ------------------------- CUDA-specific -------------------------
if (FL_LIBRARIES_USE_CUDA)
  set(FL_LIBRARIES_SEQUENCE_CRITERION_CUDA_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/cuda/CriterionUtils.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/ForceAlignmentCriterion.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/FullConnectionCriterion.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/ViterbiPath.cu
    )
else ()
  message(
    "libraries criterions are not implemented for non-CUDA backends"
    )
endif ()
