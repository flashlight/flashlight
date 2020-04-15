cmake_minimum_required(VERSION 3.5.1)
project(flashlight-vision)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(VISION_PROJECT_COMPONENT_SRC_DIR "${CMAKE_SOURCE_DIR}/vision") # module root

# If building in source, we already have these targets.
if (NOT TARGET flashlight)
  find_package(flashlight)
endif ()
if (NOT TARGET ArrayFire)
  find_package(ArrayFire)
endif ()

find_package(GFLAGS REQUIRED)

set(VISION_MODULES
  VisionDatasets
  VisionModels
)

add_subdirectory(${VISION_PROJECT_COMPONENT_SRC_DIR}/dataset)
add_subdirectory(${VISION_PROJECT_COMPONENT_SRC_DIR}/models)

set(target "Resnet34")
add_executable(${target} "Resnet34.cpp")
if(TARGET flashlight)
    target_link_libraries(${target} PRIVATE
      flashlight
      ${GFLAGS_LIBRARIES}
      ${VISION_MODULES}
    )
    target_include_directories(${target}
      PRIVATE
      ${CMAKE_SOURCE_DIR}
    )
else()
  target_link_libraries(${target} PRIVATE ${VISION_MODULES} flashlight::flashlight ${GFLAGS_LIBRARIES})
endif()
