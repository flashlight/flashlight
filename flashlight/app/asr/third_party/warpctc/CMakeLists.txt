message(STATUS "warpctc: Building with GPU support")

set(warpctc_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/src/ctc_entrypoint.cu
      ${CMAKE_CURRENT_LIST_DIR}/src/reduce.cu PARENT_SCOPE)
