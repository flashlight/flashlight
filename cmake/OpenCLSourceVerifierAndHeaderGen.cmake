# Function to turn an OpenCL source file into a C string within a source file.
# This function also calls the OpenCL driver to compile the code, thus
# providing compile time syntax verification.
#
# Usage example:
#
#  opencl_syntax_check_and_header_gen(
#     SOURCES ${kernel_src}     # opencl sources
#     VARNAME kernel_files      # cmake variable containing targets
#     EXTENSION "hpp"           # generated header file extension.
#     OUTPUT_DIR ${HEADER_DIR}  # directory for generated headers
#     TARGETS cl_kernel_targets # cmake target name
#     NAMESPACE "opencl"        # namespace that wraps the string in the file.
#     )
#
# For example, if the input file in SOURCES is "foo.cl" and
# the NAMESPACE is "opencl" the generated variables is:
#
#  namespace opencl {
#     const char* foo_cl = R("[opencl kernel code] ...");
# } // namespace opencl
#

set(SYNTAX_CHECK_AND_HEADER_GEN "opencl_syntax_and_header")

function(opencl_syntax_check_and_header_gen)
    cmake_parse_arguments(OCL "WITH_EXTENSION;NULLTERM" "VARNAME;EXTENSION;OUTPUT_DIR;TARGETS;NAMESPACE;BINARY" "SOURCES" ${ARGN})
    set(ALL_OUPUTS "")
    foreach(CUR_SRC_FILE ${OCL_SOURCES})
        get_filename_component(SRC_PATH "${CUR_SRC_FILE}" PATH)
        get_filename_component(SRC_FILE "${CUR_SRC_FILE}" NAME)
        get_filename_component(CPP_VAR "${CUR_SRC_FILE}" NAME)
        string(REPLACE "." "_" CPP_VAR ${CPP_VAR})

        set(OUPUT_DIR
          "${CMAKE_CURRENT_BINARY_DIR}/${OCL_OUTPUT_DIR}")
        set(OUPUT_FILE
          "${OUPUT_DIR}/${CPP_VAR}.${OCL_EXTENSION}")

        add_custom_command(
            OUTPUT ${OUPUT_FILE}
            DEPENDS ${CUR_SRC_FILE} ${SYNTAX_CHECK_AND_HEADER_GEN}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${OUPUT_DIR}
            COMMAND ${SYNTAX_CHECK_AND_HEADER_GEN} --input=${SRC_FILE} --output=${OUPUT_FILE} --namespace=${OCL_NAMESPACE} --var=${CPP_VAR}
            WORKING_DIRECTORY "${SRC_PATH}"
            COMMENT "Verifying OpenCL syntax of ${CUR_SRC_FILE} and compiling to C++ header ${OUPUT_FILE}."
        )
        list(APPEND ALL_OUPUTS ${OUPUT_FILE})
    endforeach()
    add_custom_target(${OCL_NAMESPACE}_${OCL_OUTPUT_DIR}_bin_target DEPENDS ${ALL_OUPUTS})
    set_target_properties(${OCL_NAMESPACE}_${OCL_OUTPUT_DIR}_bin_target PROPERTIES FOLDER "Generated Targets")

    set("${OCL_VARNAME}" ${ALL_OUPUTS} PARENT_SCOPE)
    set("${OCL_TARGETS}" ${OCL_NAMESPACE}_${OCL_OUTPUT_DIR}_bin_target PARENT_SCOPE)
endfunction(OPENCL_SYNTAX_CHECK_AND_HEADER_GEN)
