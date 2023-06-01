cmake_minimum_required(VERSION 3.16)

include(FetchContent)

# Also requires one of: libbfd (gnu binutils), libdwarf, libdw (elfutils)
FetchContent_Declare(BackwardCpp
  GIT_REPOSITORY https://github.com/bombela/backward-cpp
  GIT_TAG 2395cfa2422edb71929c9d166a6a614571331db3)

FetchContent_Populate(BackwardCpp)
FetchContent_GetProperties(BackwardCpp
  SOURCE_DIR BackwardCpp_SOURCE
  POPULATED BackwardCpp_POPULATED
  )
# Ensure Backward::Backward is defined
find_package(Backward REQUIRED CONFIG PATHS ${BackwardCpp_SOURCE})
