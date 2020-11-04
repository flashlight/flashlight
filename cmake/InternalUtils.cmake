
function(setup_install_targets)
  set(multiValueArgs INSTALL_TARGETS INSTALL_HEADERS)
  cmake_parse_arguments(setup_install_targets "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  # Main target
  install(
    TARGETS ${setup_install_targets_INSTALL_TARGETS}
    EXPORT flashlightTargets
    COMPONENT flashlight
    PUBLIC_HEADER DESTINATION fl
    RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR}
    LIBRARY DESTINATION ${FL_INSTALL_LIB_DIR}
    ARCHIVE DESTINATION ${FL_INSTALL_LIB_DIR}
    FRAMEWORK DESTINATION framework
    INCLUDES DESTINATION ${FL_INSTALL_INC_DIR}
    )

  # Write and install targets file
  install(
    EXPORT flashlightTargets
    NAMESPACE flashlight::
    DESTINATION ${FL_INSTALL_CMAKE_DIR}
    COMPONENT flashlight
    )

  # Write config file (used by projects including fl, such as examples)
  include(CMakePackageConfigHelpers)
  set(INCLUDE_DIRS include)
  set(CMAKE_DIR ${FL_INSTALL_CMAKE_DIR})
  configure_package_config_file(
    ${CMAKE_MODULE_PATH}/flashlightConfig.cmake.in
    cmake/install/${FL_CONFIG_CMAKE_BUILD_DIR}/flashlightConfig.cmake
    INSTALL_DESTINATION
    ${FL_INSTALL_CMAKE_DIR}
    PATH_VARS INCLUDE_DIRS CMAKE_DIR
    )
  install(FILES
    ${PROJECT_BINARY_DIR}/cmake/install/flashlightConfig.cmake
    DESTINATION ${FL_INSTALL_CMAKE_DIR}
    COMPONENT flashlight
    )
endfunction(setup_install_targets)

function(setup_install_headers HEADER_DIR DEST_DIR)
  # Move headers
  install(
    DIRECTORY ${HEADER_DIR}
    COMPONENT headers
    DESTINATION ${DEST_DIR}
    FILES_MATCHING # preserve directory structure
    PATTERN  "*.h"
    PATTERN "*.cuh" # TODO: gate me
    PATTERN "test*" EXCLUDE
    PATTERN "tests" EXCLUDE
    PATTERN "backend" EXCLUDE
    PATTERN "examples" EXCLUDE
    PATTERN "third_party" EXCLUDE
    PATTERN "experimental" EXCLUDE
    )
endfunction(setup_install_headers)
