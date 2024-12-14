#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenCL::UtilsCpp" for configuration "Release"
set_property(TARGET OpenCL::UtilsCpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OpenCL::UtilsCpp PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libOpenCLUtilsCpp.dll.a"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libOpenCLUtilsCpp.dll"
  )

list(APPEND _cmake_import_check_targets OpenCL::UtilsCpp )
list(APPEND _cmake_import_check_files_for_OpenCL::UtilsCpp "${_IMPORT_PREFIX}/lib/libOpenCLUtilsCpp.dll.a" "${_IMPORT_PREFIX}/bin/libOpenCLUtilsCpp.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
