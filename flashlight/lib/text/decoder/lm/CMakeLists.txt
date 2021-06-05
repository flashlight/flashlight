cmake_minimum_required(VERSION 3.10)

set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum ngram order for KenLM")

target_sources(
  fl-libraries
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ConvLM.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ZeroLM.cpp
  )

# ------------------------- KenLM-specific -------------------------

if (FL_LIBRARIES_USE_KENLM)
  # Try to find the CMake config first
  find_package(kenlm CONFIG)
  if (NOT kenlm_FOUND)
    # No luck - use Findkenlm.cmake
    find_package(kenlm MODULE)
    if (NOT kenlm_FOUND)
      if (FL_BUILD_STANDALONE)
        message(STATUS "KenLM not found - will download and build from source")
        include(${CMAKE_MODULE_PATH}/BuildKenlm.cmake)
        add_dependencies(fl-libraries kenlm) # ExternalProject target
      endif()
    endif()
  endif()

  target_sources(
    fl-libraries
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/KenLM.cpp
    )

  target_compile_definitions(
    fl-libraries
    PUBLIC
    KENLM_MAX_ORDER=${KENLM_MAX_ORDER}
    )

  # Link imported kenlm target if we have it
  if (TARGET kenlm::kenlm AND TARGET kenlm::kenlm_util)
    target_link_libraries(
      fl-libraries
      PRIVATE
      kenlm::kenlm
      kenlm::kenlm_util
      )
  else()
    target_link_libraries(
      fl-libraries
      PRIVATE
      ${KENLM_LIBRARIES}
      )

    target_include_directories(
      fl-libraries
      PUBLIC
      $<BUILD_INTERFACE:${KENLM_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${KENLM_INCLUDE_DIRS_LM}>
      )
  endif()
endif()
