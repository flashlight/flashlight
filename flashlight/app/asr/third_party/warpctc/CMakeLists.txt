cmake_minimum_required(VERSION 3.5.1)

project(ctc_release)

#find_package(CUDA 6.5)
#find_package(OpenMP)
#find_package(Torch)

#option(WITH_GPU     "compile warp-ctc with CUDA."     ${CUDA_FOUND})
#option(WITH_OMP     "compile warp-ctc with OpenMP."   ${OpenMP_FOUND})
#option(WITH_TORCH   "compile warp-ctc with Torch."    ${Torch_FOUND})
#option(BUILD_TESTS  "build warp-ctc unit tests."      ON)
#option(BUILD_SHARED "build warp-ctc shared library."  ON)

#set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build Type of the project.")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)


if(WIN32)
    set(CMAKE_STATIC_LIBRARY_PREFIX lib)
endif(WIN32)

# Apple specifics
IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    add_compile_definitions(APPLE)

    #for el capitain have to use rpath
    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


# windows treat symbolic file as a real file, which is different with unix
# We create a hidden file and compile it instead of origin source file.
function(windows_symbolic TARGET)
    set(oneValueArgs "")
    set(multiValueArgs SRCS PATH DEPS)
    cmake_parse_arguments(windows_symbolic "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(final_path ${CMAKE_CURRENT_SOURCE_DIR}/${windows_symbolic_PATH})
    foreach(src ${windows_symbolic_SRCS})
        get_filename_component(src ${src} NAME_WE)
        if (NOT EXISTS ${final_path}/${src}.cpp OR NOT EXISTS ${final_path}/${src}.cu)
            message(FATAL " ${final_path}/${src}.cc and ${final_path}/${src}.cu must exsits, and ${final_path}/${src}.cu must be symbolic file.")
        endif()

        # only copy the xx.cu to .xx.cu when the content are modified
        set(copy_flag 1)
        if (EXISTS ${final_path}/.${src}.cu)
            file(READ ${final_path}/${src}.cpp SOURCE_STR)
            file(READ ${final_path}/.${src}.cu TARGET_STR)
            if (SOURCE_STR STREQUAL TARGET_STR)
                set(copy_flag 0)
            endif()
        endif()
        if (copy_flag)
            add_custom_command(OUTPUT ${final_path}/.${src}.cu
                    COMMAND ${CMAKE_COMMAND} -E remove ${final_path}/.${src}.cu
                    COMMAND ${CMAKE_COMMAND} -E copy "${final_path}/${src}.cpp" "${final_path}/.${src}.cu"
                    COMMENT "create hidden file of ${src}.cu")
        endif()
        add_custom_target(${TARGET} ALL DEPENDS ${final_path}/.${src}.cu)
    endforeach()
endfunction()

#if(NOT WITH_GPU)
    #message(STATUS "warpctc: Building shared library with no GPU support")
    #add_library(warpctc src/ctc_entrypoint.cpp)
#else()
    message(STATUS "warpctc: Building shared library with GPU support")

    # need to be at least 30 or __shfl_down in reduce wont compile
    IF (CUDA_VERSION VERSION_LESS "11.0")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
    ENDIF()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

    if (CUDA_VERSION VERSION_GREATER "7.6")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
    endif()

    if ((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    endif()

    if ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    endif()

    if ((CUDA_VERSION VERSION_GREATER "11.0") OR (CUDA_VERSION VERSION_EQUAL "11.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    endif()

    if (WIN32)
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd 4068 /wd 4244 /wd 4267 /wd 4305 /wd 4819\"")
        windows_symbolic(ctc_entrypoint SRCS ctc_entrypoint.cu PATH src)
        cuda_add_library(warpctc src/.ctc_entrypoint.cu src/reduce.cu)
    else()
        cuda_add_library(warpctc src/ctc_entrypoint.cu src/reduce.cu)
    endif()
#endif()


#if(WITH_OMP)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
else()
    add_compile_definitions(CTC_DISABLE_OMP)
endif()

target_include_directories(warpctc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
#install(TARGETS warpctc)
#install(FILES include/ctc.h DESTINATION "include")


# If Torch is present
#if (WITH_TORCH)
    #if(WITH_GPU)
        #message(STATUS "Building Torch Bindings with GPU upport")
    #else()
        #message(STATUS "Building Torch Bindings with no GPU support")
    #endif()

    #target_link_libraries(warpctc PUBLIC luajit luaT TH)
    #target_include_directories(${Torch_INSTALL_INCLUDE} ${Torch_INSTALL_INCLUDE}/TH)

    #install(TARGETS warpctc
            #RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
            #LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
            #ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

    #add_torch_package(
    #warp_ctc
        #torch_binding/binding.cpp
        #torch_binding/utils.c
        #torch_binding/init.lua
    #)
    #target_link_libraries(warp_ctc warpctc)

    #if (NOT APPLE)
        #target_link_libraries(warp_ctc gomp)
    #endif()

    #if(WITH_GPU)
        #include_directories(${Torch_INSTALL_INCLUDE}/THC ${CUDA_INCLUDE_DIRS} "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")
        #target_link_libraries(warpctc THC ${CUDA_curand_LIBRARY})
        #target_link_libraries(warp_ctc THC ${CUDA_curand_LIBRARY})
    #else()
        #add_compile_definitions(TORCH_NOGPU)
    #endif()

#endif()


# Build Tests
#if(BUILD_TESTS)
    #add_executable(test_cpu tests/test_cpu.cpp )
    #target_link_libraries(test_cpu warpctc)
    #target_include_directories(test_cpu PRIVATE tests)

    #if(WITH_GPU)
        #cuda_add_executable(test_gpu tests/test_gpu.cu)
        #target_link_libraries(test_gpu warpctc ${CUDA_curand_LIBRARY})
        #target_include_directories(test_gpu PRIVATE tests)
    #endif()

#endif()
