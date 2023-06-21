cmake_minimum_required(VERSION 3.16)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DevicePtr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DynamicBenchmark.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Histogram.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Timer.cpp
)

if (${FL_BUILD_PLUGIN})
  target_sources(flashlight PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Plugin.cpp)
else()
  message(WARNING "Compiler doesn't support -rdynamic - not building Plugin")
endif()

# A native threading library
find_package(Threads REQUIRED)
target_link_libraries(flashlight PUBLIC Threads::Threads)

# Dynamic lib loading needed for Plugins
target_link_libraries(flashlight PUBLIC ${CMAKE_DL_LIBS})

# To support __VA_ARGS__ for serialization macros
if (MSVC)
  if (MSVC_VERSION LESS 1920)
    target_compile_options(flashlight PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:/experimental:preprocessor>) # MSVC 2017
  else()
    target_compile_options(flashlight PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>) # MSVC 2019+
  endif()
endif()

# std::filesystem linking -- remove this when requiring gcc >= 9
if (NOT MSVC)
  find_package(Filesystem REQUIRED COMPONENTS Final Experimental)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindFilesystem.cmake)
  target_link_libraries(flashlight PUBLIC std::filesystem)
endif()

# backward
include(${CMAKE_CURRENT_LIST_DIR}/stacktrace/CMakeLists.txt)
