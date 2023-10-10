cmake_minimum_required(VERSION 3.16)

set(FL_PKG_DIR ${CMAKE_CURRENT_LIST_DIR})

function(register_package)
  set(oneValueArgs NAME)
  set(multiValueArgs DEPENDENT_OPTS)
  cmake_parse_arguments(register_package "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  set(_pkg_name ${register_package_NAME})
  string(TOLOWER ${_pkg_name} _pkg_name_lower)
  string(TOUPPER ${_pkg_name} _pkg_name_upper)

  set(_option_name "FL_BUILD_PKG_${_pkg_name_upper}")
  fl_dependent_option(
    OPTION ${_option_name}
    TEXT "Build ${_pkg_name_lower} pkg for Flashlight"
    VAL OFF
    DEPS "${register_package_DEPENDENT_OPTS}"
    FRCE OFF
  )

  if(${_option_name})
    if (MSVC)
      message(FATAL_ERROR "Building fl packages (pkg) with MSVC not yet supported")
    endif()

    message(STATUS "Building Flashlight ${_pkg_name_lower} package.")
    include(${FL_PKG_DIR}/${_pkg_name_lower}/CMakeLists.txt)
  endif()
endfunction()

register_package(NAME runtime DEPENDENT_OPTS FL_BUILD_CONTRIB)
register_package(NAME vision  DEPENDENT_OPTS FL_BUILD_CONTRIB)
register_package(NAME text    DEPENDENT_OPTS FL_BUILD_CONTRIB)
register_package(NAME speech  DEPENDENT_OPTS FL_BUILD_CONTRIB FL_BUILD_PKG_RUNTIME)
