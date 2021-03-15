# defines the utility function
# opencl_add_kernels_in_dir_to_target(DIRECTORY dir TARGET target)
# Steps:
# - Builds OpenCLSourceVerifierAndHeaderGen executable that generate c++
#   header files from OpenCL source code and compile them code, using
#   the OpenCL driver, to verify syntax at compile time.
# - Includes function definition
# - Setup compiler definitions.

include(OpenCLSourceVerifierAndHeaderGen)

# Arguments:
# DIRECTORY off opencl source files. OpenCL source filers are filitered using
#  the .cl file extentions
# TARGET to add the opencl code to.
function(opencl_add_kernels_in_dir_to_target)
  set(options)
  set(oneValueArgs DIRECTORY TARGET)
  set(multiValueArgs)
  cmake_parse_arguments(opencl_add_kernels_in_dir_to_target "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  file(GLOB kernel_src ${opencl_add_kernels_in_dir_to_target_DIRECTORY}/*.cl)
  set(OPENCL_KERNEL_DIR "opencl_kernels")

  opencl_syntax_check_and_header_gen(
      SOURCES ${kernel_src}
      VARNAME kernel_files
      EXTENSION "h"  # extension to the header file contining opencl source.
      OUTPUT_DIR ${OPENCL_KERNEL_DIR}
      TARGETS cl_kernel_targets
      NAMESPACE "opencl" # namespace that wraps the string in the file.
      )

  target_sources(
    ${opencl_add_kernels_in_dir_to_target_TARGET}
    PRIVATE
    ${kernel_files}
    )

  target_include_directories(
    ${opencl_add_kernels_in_dir_to_target_TARGET}
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_dependencies(${opencl_add_kernels_in_dir_to_target_TARGET} ${cl_kernel_targets})

  set(opencl_compile_definitions
        CL_TARGET_OPENCL_VERSION=120
        CL_HPP_TARGET_OPENCL_VERSION=120
        CL_HPP_MINIMUM_OPENCL_VERSION=120
        CL_HPP_ENABLE_EXCEPTIONS
        CL_USE_DEPRECATED_OPENCL_1_2_APIS
  )

  target_compile_definitions(
    ${opencl_add_kernels_in_dir_to_target_TARGET}
    PRIVATE
    ${opencl_compile_definitions}
    )

endfunction(opencl_add_kernels_in_dir_to_target)
