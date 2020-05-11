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

add_library(flashlightVision "")

set(VISION_MODULES
  VisionDatasets
  VisionModels
  VisionCriterion
)

target_link_libraries(
  flashlightVision
  PUBLIC # export dependency library and include paths for each module
  ${VISION_MODULES}
  flashlight
  )

# Internal includes are impl defined as <flashlight...>
target_include_directories(
  flashlightVision
  PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_subdirectory(${VISION_PROJECT_COMPONENT_SRC_DIR}/dataset)
add_subdirectory(${VISION_PROJECT_COMPONENT_SRC_DIR}/models)
add_subdirectory(${VISION_PROJECT_COMPONENT_SRC_DIR}/criterion)



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

function(build_train SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
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
endfunction(build_train)

build_train(Detr.cpp)

enable_testing()
add_subdirectory(${VISION_PROJECT_COMPONENT_SRC_DIR}/tests)
