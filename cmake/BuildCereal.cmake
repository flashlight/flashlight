cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
  cereal
  GIT_REPOSITORY https://github.com/USCiLab/cereal.git
  GIT_TAG        v1.3.2
  )

# Save
set(_BUILD_SANDBOX ${BUILD_SANDBOX})
set(_BUILD_DOC ${BUILD_DOC})

set(BUILD_SANDBOX OFF CACHE INTERNAL "Disable building cereal sandbox")
set(BUILD_DOC OFF CACHE INTERNAL "Disable building cereal docs")
set(SKIP_PERFORMANCE_COMPARISON ON CACHE INTERNAL "Skip perf comparison in cereal")
set(CEREAL_INSTALL ON CACHE INTERNAL "Force cereal install step if needed")

FetchContent_MakeAvailable(cereal)

# Restore
set(BUILD_SANDBOX ${_BUILD_SANDBOX})
set(BUILD_DOC ${_BUILD_DOC})
