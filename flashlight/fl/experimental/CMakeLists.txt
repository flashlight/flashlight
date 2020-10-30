cmake_minimum_required(VERSION 3.5.1)

project(flashlight-experimental)

# Experimental projects to build.
# To add your experimental project:
# - Add a subdirectory in `src/experimental`
# - Add a CMakeLists.txt file in that directory which builds your targets
#   (this will be automatically linked to the rest of flashlight, but can
#    also be linked to other dependencies)
# - Add a call to `include` below for your experimental
#   projects directory - for example:
#
#   include(${CMAKE_CURRENT_LIST_DIR}/[some_project_dir]/CMakeLists.txt)
#
# Add includes here:


# Memory
include(${CMAKE_CURRENT_LIST_DIR}/memory/CMakeLists.txt)

# Tests
include(${CMAKE_CURRENT_LIST_DIR}/tests/CMakeLists.txt)

# - Also add your project's target name `FLASHLIGHT_EXPERIMENTAL_LIBS`
#   call below if you want your code to be installed with flashlight.
# - Note that experimental targets MUST be `INTERFACE` targets for
#   them to be installable with flashlight.
set(
  FLASHLIGHT_EXPERIMENTAL_LIBS
  # Experimental project target names go here
  experimental_memory
  )

# Root dir is `flashlight/fl/experimental`
set(FL_EXPERIMENTAL_ROOT_PATH
    ${FLASHLIGHT_PROJECT_COMPONENT_SRC_DIR}/experimental)

add_library(
  FlExperimental
  INTERFACE
  )

target_link_libraries(
  FlExperimental
  INTERFACE
  ${FLASHLIGHT_EXPERIMENTAL_LIBS}
  )
