
function(setup_install)
  set(multiValueArgs INSTALL_TARGETS INSTALL_HEADERS)
  cmake_parse_arguments(setup_install "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})
    
  # Main target
  install(
    TARGETS ${setup_install_INSTALL_TARGETS}
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
    EXPORT
    flashlightTargets
    NAMESPACE
    flashlight::
    DESTINATION
    ${FL_INSTALL_CMAKE_DIR}
    COMPONENT
    cmake
    )

  # Move headers
  install(
    DIRECTORY
    ${setup_install_INSTALL_HEADERS}
    COMPONENT
    headers
    DESTINATION
    ${FL_INSTALL_INC_DIR_HEADER_LOC}
    FILES_MATCHING # preserve directory structure
    PATTERN  "*.h"
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
    COMPONENT cmake
    )
endfunction(setup_install)
