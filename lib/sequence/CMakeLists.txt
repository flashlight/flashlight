cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Components -------------------------

# criterion-library
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

set(FL_LIBRARIES_SEQUENCE_CUDA_SOURCES
  ${FL_LIBRARIES_SEQUENCE_CRITERION_CUDA_SOURCES}
  )
 
