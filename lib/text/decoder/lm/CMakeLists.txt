cmake_minimum_required(VERSION 3.5.1)

set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum ngram order for KenLM")

target_sources(
  fl-libraries
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ConvLM.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ZeroLM.cpp
  )

# ------------------------- KenLM-specific -------------------------

if (FL_LIBRARIES_USE_KENLM)
  find_package(kenlm REQUIRED)

  # Required for KenLM to read ARPA files in compressed format
  find_package(LibLZMA)
  find_package(BZip2)
  find_package(ZLIB)

  target_sources(
    fl-libraries
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/KenLM.cpp    
    )

  target_link_libraries(
    fl-libraries
    PUBLIC
    ${KENLM_LIBRARIES}
    ${LIBLZMA_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${ZLIB_LIBRARIES}
    )

  target_include_directories(
    fl-libraries
    PUBLIC
    ${KENLM_INCLUDE_DIRS}
    ${LIBLZMA_INCLUDE_DIRS}
    ${BZIP2_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
    )

  target_compile_definitions(
    fl-libraries
    PUBLIC
    KENLM_MAX_ORDER=${KENLM_MAX_ORDER}
    )
endif ()
