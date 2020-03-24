cmake_minimum_required(VERSION 3.5.1)

project(flashlight-experimental)

# Experimental projects to build.
# To add your experimental project:
# - Add a subdirectory in `src/experimental`
# - Add a CMakeLists.txt file in that subdirectory which builds your targets
#   (this target can link to other dependencies, e.g. flashlight::flashlight)
# - Add a call to add_subdirectory below for your experimental
#   projects directory - for example:
#
#   add_subdirectory
#
# - Also add your project's target name `FLASHLIGHT_EXPERIMENTAL_LIBS`
#   call below if you want your code to be installed with flashlight.
set(
  FLASHLIGHT_EXPERIMENTAL_LIBS
  # Experimental project target names go here
  )

# Root dir is `flashlight/experimental`
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
