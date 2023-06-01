cmake_minimum_required(VERSION 3.16)

set(BACKWARD_SHARED ${BUILD_SHARED_LIBS})

include(FetchContent)

# Also requires one of: libbfd (gnu binutils), libdwarf, libdw (elfutils)
FetchContent_Declare(BackwardCpp
        GIT_REPOSITORY https://github.com/bombela/backward-cpp
        GIT_TAG 2395cfa2422edb71929c9d166a6a614571331db3)
FetchContent_MakeAvailable(BackwardCpp)
