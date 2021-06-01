cmake_minimum_required(VERSION 3.10)

include(${CMAKE_MODULE_PATH}/Buildpybind11.cmake)
include(${CMAKE_MODULE_PATH}/pybind11Tools.cmake)

set_target_properties(
  fl_lib_text
  PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE
  TRUE
  )

function (add_pybind11_extension ext_name)
  string(REPLACE "_" ";" modlist ${ext_name})
  list(GET modlist -1 modname)
  list(REMOVE_AT modlist -1)
  if(modlist)
    string(REPLACE ";" "/" relpath "${modlist}")
  else()
    set(relpath "")
  endif()

  pybind11_add_module(
    ${ext_name}
    ${CMAKE_CURRENT_LIST_DIR}/${relpath}/_${modname}.cpp
    )

  target_link_libraries(
    ${ext_name}
    PUBLIC
    fl_lib_text
    fl_lib_audio
    fl_lib_sequence
    fl_lib_common
    )

  add_dependencies(${ext_name} pybind11)
  set_target_properties(${ext_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${relpath})
  set_target_properties(${ext_name} PROPERTIES OUTPUT_NAME ${ext_name})

endfunction ()

add_pybind11_extension(flashlight_lib_audio_feature)
add_pybind11_extension(flashlight_lib_sequence_criterion)
add_pybind11_extension(flashlight_lib_text_decoder)
add_pybind11_extension(flashlight_lib_text_dictionary)
