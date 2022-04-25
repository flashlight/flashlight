cmake_minimum_required(VERSION 3.10)

# ----------------------------- Common -----------------------------
set(
  COMMON_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DevicePtr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DynamicBenchmark.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Histogram.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Plugin.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Timer.cpp
)

target_sources(
  flashlight
  PRIVATE
  ${COMMON_SRCS}
)

# A native threading library is needed for ThreadPool
# (see flashlight/common/threadpool/ThreadPool.h)
find_package(Threads REQUIRED)

target_link_libraries(
  flashlight
  PUBLIC
  ${CMAKE_THREAD_LIBS_INIT} # threading library
  ${CMAKE_DL_LIBS} # for plugin
)

# Remove this when requiring gcc >= 9
find_package(Filesystem REQUIRED COMPONENTS Final Experimental)
setup_install_find_module(${CMAKE_MODULE_PATH}/FindFilesystem.cmake)
target_link_libraries(flashlight PUBLIC std::filesystem)
