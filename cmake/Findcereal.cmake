# Try to find Cereal
#
# Sets the following variables:
# CEREAL_FOUND
# CEREAL_INCLUDE_DIRS - directories with Cereal headers
# CEREAL_DEFINITIONS - Cereal compiler flags

find_path(CEREAL_INCLUDE_DIR
  cereal
	HINTS
    "$ENV{CEREAL_ROOT}/include"
    "/usr/include"
    "$ENV{PROGRAMFILES}/cereal/include"
)

set(CEREAL_INCLUDE_DIRS ${CEREAL_INCLUDE_DIR})

if (CEREAL_INCLUDE_DIRS)
  set(cereal_FOUND TRUE)
  message(STATUS "cereal found (include: ${CEREAL_INCLUDE_DIRS})")
else()
  set(cereal_FOUND TRUE)
  message(STATUS "cereal not found")
endif()
