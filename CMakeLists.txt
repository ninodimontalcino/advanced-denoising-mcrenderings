cmake_minimum_required (VERSION 3.10)

# #########################################################################
# DEFINE PROJECT
project(team053)


set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wno-deprecated -O3 -ffast-math -march=native" )

###########################################################################
# OpenEXR (taken from https://github.com/mmp/pbrt-v3/blob/master/CMakeLists.txt)

SET(ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
SET(OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
SET(OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
SET(ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)

ADD_SUBDIRECTORY(src/ext/openexr)

SET_PROPERTY(TARGET IexMath eLut toFloat b44ExpLogTable dwaLookups IlmThread Half Iex Imath IlmImf PROPERTY FOLDER "ext")

INCLUDE_DIRECTORIES (
  src/ext/openexr/IlmBase/Imath
  src/ext/openexr/IlmBase/Half
  src/ext/openexr/IlmBase/Iex
  src/ext/openexr/OpenEXR/IlmImf
  ${CMAKE_BINARY_DIR}/src/ext/openexr/IlmBase/config
  ${CMAKE_BINARY_DIR}/src/ext/openexr/OpenEXR/config
)
IF(WIN32)
  SET(OPENEXR_LIBS IlmImf Imath Half ${ZLIB_LIBRARY})
ELSE()
  SET(OPENEXR_LIBS IlmImf Imath Half)
ENDIF()

###########################################################################
# EXECUTABLES
# -------------------------------------------------------------------------

# --------------------------
# (1) Main Executable
# --------------------------
add_executable(main 

              # Header Files
              tsc_x86.h 
              exr.h
              memory_mgmt.hpp
              denoise.h
              flt.hpp 
              flt_restructure.hpp
              validation.hpp

              # General Source Files
              main.cpp 
              exr.cpp
              memory_mgmt.cpp 
              flt.cpp 
              flt_restructure.cpp 
              flt_restructure_ILP.cpp 
              validation.cpp

              # Implementations (BASIC and OPTIMIZED)
              implementations/register.hpp
              #implementations/basic.cpp
              #implementations/basic_restructure.cpp
              implementations/basic_restructure2.cpp
              implementations/basic_restructure3.cpp

              )

# --------------------------
# (2) Testing Executable
# --------------------------
#add_executable(denoise_test flt.hpp tests.hpp memory_mgmt.hpp flt.cpp tests.cpp memory_mgmt.cpp )


# Link to dependency libraries (e.g. OPENEXR)
target_link_libraries(main ${OPENEXR_LIBS})

