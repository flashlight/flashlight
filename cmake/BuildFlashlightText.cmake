cmake_minimum_required(VERSION 3.10.0)

include(ExternalProject)

set(flashlight-text_TEMP_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern/flashlight-text)
set(flashlight-text_URL https://github.com/flashlight/text.git)
set(flashlight-text_BUILD ${CMAKE_CURRENT_BINARY_DIR}/third-party/flashlight-text)
set(flashlight-text_TAG 46f62b29ec2db8389efcdb731ebff05fb400a95e) # 20220928
set(flashlight-text_BINARY_DIR ${flashlight-text_BUILD}/src/flashlight-text-build)

if (BUILD_SHARED_LIBS)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

set(FLASHLIGHT_TEXT_LIB_PATH ${flashlight-text_BINARY_DIR}/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}flashlight-text${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX})

if (NOT TARGET flashlight-text)
  # Download flashlight-text
  ExternalProject_Add(
    flashlight-text
    PREFIX ${flashlight-text_BUILD}
    GIT_REPOSITORY ${flashlight-text_URL}
    GIT_TAG ${flashlight-text_TAG}
    BUILD_BYPRODUCTS
      ${FLASHLIGHT_TEXT_LIB_PATH}
    CMAKE_CACHE_ARGS
      -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${flashlight-text_TEMP_INSTALL_DIR}
      -DFL_TEXT_BUILD_TESTS:BOOL=OFF
      -DFL_TEXT_BUILD_STANDALONE:BOOL=OFF
      # TODO: flashlight-text doesn't currently work with KenLM CMake
      # ExternalProject -- make this better
      -DFL_TEXT_USE_KENLM:BOOL=${FL_BUILD_APP_ASR}
      )
endif ()

# Install the install executed at build time
install(DIRECTORY ${flashlight-text_TEMP_INSTALL_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${flashlight-text_TEMP_INSTALL_DIR}/lib DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${flashlight-text_TEMP_INSTALL_DIR}/share DESTINATION ${CMAKE_INSTALL_PREFIX})

set(FLASHLIGHT_TEXT_INCLUDE_DIRS ${flashlight-text_TEMP_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${FLASHLIGHT_TEXT_INCLUDE_DIRS})

if (NOT TARGET flashlight::flashlight-text)
  add_library(flashlight::flashlight-text ${LIB_TYPE} IMPORTED)
  set_property(TARGET flashlight::flashlight-text PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FLASHLIGHT_TEXT_INCLUDE_DIRS})
  set_property(TARGET flashlight::flashlight-text PROPERTY IMPORTED_LOCATION ${FLASHLIGHT_TEXT_LIB_PATH})
  add_dependencies(flashlight::flashlight-text flashlight-text)
endif()

if (${FL_BUILD_APP_ASR})
  # Bundle KenLM libraries into the flashlight-text imported target if usin the Flashlight ASR lib
  # which requires KenLM
  find_package(kenlm REQUIRED)
  set_property(TARGET flashlight::flashlight-text PROPERTY
    IMPORTED_LINK_INTERFACE_LIBRARIES kenlm::kenlm kenlm::kenlm_util)
endif()
