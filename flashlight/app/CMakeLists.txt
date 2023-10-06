cmake_minimum_required(VERSION 3.16)

set(FL_APPS_DIR ${CMAKE_CURRENT_LIST_DIR})

function(register_app)
  set(oneValueArgs NAME)
  set(multiValueArgs DEPENDENT_OPTS)
  cmake_parse_arguments(register_app "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  set(_app_name ${register_app_NAME})
  string(TOLOWER ${_app_name} _app_name_lower)
  string(TOUPPER ${_app_name} _app_name_upper)

  set(_option_name "FL_BUILD_APP_${_app_name_upper}")
  fl_dependent_option(
    OPTION ${_option_name}
    TEXT "Build ${_app_name_lower} app for Flashlight"
    VAL OFF
    DEPS "${register_app_DEPENDENT_OPTS}"
    FRCE OFF
  )

  if(${_option_name})
    if (MSVC)
      message(FATAL_ERROR "Building fl packages (pkg) with MSVC not yet supported")
    endif()

    message(STATUS "Building Flashlight ${_app_name_lower} package.")
    include(${FL_APPS_DIR}/${_app_name_lower}/CMakeLists.txt)
  endif()
endfunction()

set(_common_deps FL_BUILD_CONTRIB FL_BUILD_PKG_RUNTIME) # for all apps
register_app(NAME asr       DEPENDENT_OPTS ${_common_deps} FL_BUILD_PKG_SPEECH)
register_app(NAME imgclass  DEPENDENT_OPTS ${_common_deps} FL_BUILD_PKG_VISION)
register_app(NAME objdet    DEPENDENT_OPTS ${_common_deps} FL_BUILD_PKG_VISION)
register_app(NAME lm        DEPENDENT_OPTS ${_common_deps} FL_BUILD_PKG_TEXT)
register_app(
  NAME benchmark
  DEPENDENT_OPTS
  ${_common_deps}
  FL_BUILD_PKG_SPEECH
  FL_BUILD_PKG_VISION
  FL_BUILD_PKG_TEXT
)
