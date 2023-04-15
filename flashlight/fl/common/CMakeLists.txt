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
  ${CMAKE_CURRENT_LIST_DIR}/Plugin.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Timer.cpp
)

# A native threading library
find_package(Threads REQUIRED)
target_link_libraries(flashlight PUBLIC Threads::Threads)

# Dynamic lib loading needed for Plugins
target_link_libraries(flashlight PUBLIC ${CMAKE_DL_LIBS})

# To support __VA_ARGS__ for serialization macros
if (MSVC)
  if (MSVC_VERSION LESS 1920)
    target_compile_options(flashlight PUBLIC "/experimental:preprocessor") # MSVC 2017
  else()
    target_compile_options(flashlight PUBLIC "/Zc:preprocessor") # MSVC 2019+
  endif()
endif()

# std::filesystem -- remove this when requiring gcc >= 9
# find_package(Filesystem REQUIRED COMPONENTS Final Experimental)
# setup_install_find_module(${CMAKE_MODULE_PATH}/FindFilesystem.cmake)
# target_link_libraries(flashlight PUBLIC std::filesystem)
