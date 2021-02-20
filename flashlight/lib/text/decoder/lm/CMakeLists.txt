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
  find_package(kenlm)
  if (NOT kenlm_FOUND)
    if (FL_BUILD_STANDALONE)
      message(STATUS "KenLM not found - will download and build from source")
      include(${CMAKE_MODULE_PATH}/BuildKenlm.cmake)
      add_dependencies(fl-libraries kenlm) # ExternalProject target
    endif()
  endif()
  if (FL_BUILD_STANDALONE)
    setup_install_find_module(${CMAKE_MODULE_PATH}/Findkenlm.cmake)
  endif()

  target_sources(
    fl-libraries
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/KenLM.cpp
    )

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

  target_compile_definitions(
    fl-libraries
    PUBLIC
    KENLM_MAX_ORDER=${KENLM_MAX_ORDER}
    )
endif ()
