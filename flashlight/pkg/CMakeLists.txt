cmake_minimum_required(VERSION 3.16)

function(register_package)
  set(oneValueArgs NAME)
  set(multiValueArgs DEPENDENT_OPTS)
  cmake_parse_arguments(register_package "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  set(_pkg_name ${register_package_NAME})
  string(TOLOWER ${_pkg_name} _pkg_name_lower)

  set(_option_name ${FL_BUILD_PKG_${_pkg_name})
  fl_dependent_option(
    ${_option_name}
    "Build ${_pkg_name_lower} package for Flashlight"
    OFF
    "${register_package_DEPENDENT_OPTS}"
    OFF
  )

  if(${_option_name})
    message(STATUS "Building Flashlight ${_pkg_name_lower} package.")
    include(${FL_PKG_DIR}/${_pkg_name_lower})
  endif()
endfunction()

register_package(NAME runtime DEPENDENT_OPTS FL_BUILD_CONTRIB)
register_package(NAME vision  DEPENDENT_OPTS FL_BUILD_CONTRIB)
register_package(NAME text    DEPENDENT_OPTS FL_BUILD_CONTRIB)
register_package(NAME speech  DEPENDENT_OPTS FL_BUILD_CONTRIB FL_BUILD_PKG_RUNTIME)
