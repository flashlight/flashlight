cmake_minimum_required(VERSION 3.5.1)

add_library(
  flashlight-app-img-class
  ""
  )

# Find GLog
find_package(GLOG REQUIRED)
if (GLOG_FOUND)
  message(STATUS "GLOG found")
else()
  message(FATAL_ERROR "GLOG not found")
endif()

# Find GFlags
find_package(GFLAGS REQUIRED)
if (GFLAGS_FOUND)
  message(STATUS "GFLAGS found")
else()
  message(FATAL_ERROR "GFLAGS not found")
endif()

if (FL_LIBRARIES_USE_MKL)
  find_package(MKL)
  if (NOT MKL_FOUND)
    message(WARNING "MKL not found; forcing FL_LIBRARIES_USE_MKL=OFF.")
    set(FL_LIBRARIES_USE_MKL OFF CACHE BOOL "" FORCE)
  endif ()
endif ()

target_link_libraries(
  flashlight-app-img-class
  PUBLIC
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-img-class
  PUBLIC
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)


add_executable(Resnet34 ${CMAKE_CURRENT_LIST_DIR}/examples/Resnet34.cpp)
#target_link_libraries(Resnet34 flashlight-app-vision)
target_link_libraries(Resnet34 flashlight-app-img-class)

add_executable(Resnet34Backbone ${CMAKE_CURRENT_LIST_DIR}/examples/Resnet34Backbone.cpp)
target_link_libraries(Resnet34Backbone flashlight-app-img-class)



