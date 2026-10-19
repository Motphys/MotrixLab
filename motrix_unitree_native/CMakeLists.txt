cmake_minimum_required(VERSION 3.18)
project(motrix_unitree_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(UNITREE_SDK2_ROOT "" CACHE PATH "Unitree SDK2 source or installation root")
find_package(unitree_sdk2 CONFIG QUIET)

if(NOT unitree_sdk2_FOUND)
  if(NOT UNITREE_SDK2_ROOT)
    if(DEFINED ENV{UNITREE_SDK2_ROOT})
      set(UNITREE_SDK2_ROOT "$ENV{UNITREE_SDK2_ROOT}")
    else()
      set(UNITREE_SDK2_ROOT "/opt/unitree_robotics")
    endif()
  endif()

  if(NOT EXISTS "${UNITREE_SDK2_ROOT}/include/unitree/robot/channel/channel_factory.hpp")
    # CI and simulation-only installs must still resolve `--extra unitree`; the
    # adapter imports the extension lazily and reports the missing backend.
    message(WARNING
      "Unitree SDK2 headers were not found under ${UNITREE_SDK2_ROOT}; building "
      "motrix_unitree_native without the native DDS backend. Set UNITREE_SDK2_ROOT "
      "and reinstall to enable hardware transport.")
    return()
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(UNITREE_ARCH "x86_64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(UNITREE_ARCH "aarch64")
  else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  set(UNITREE_SDK2_LIBRARY "${UNITREE_SDK2_ROOT}/lib/${UNITREE_ARCH}/libunitree_sdk2.a")
  set(UNITREE_DDS_LIBRARY "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}/libddsc.so")
  set(UNITREE_DDSXX_LIBRARY "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}/libddscxx.so")
  if(NOT EXISTS "${UNITREE_SDK2_LIBRARY}" OR NOT EXISTS "${UNITREE_DDS_LIBRARY}" OR
     NOT EXISTS "${UNITREE_DDSXX_LIBRARY}")
    message(FATAL_ERROR
      "Unitree SDK2 libraries were not found for ${UNITREE_ARCH} under ${UNITREE_SDK2_ROOT}")
  endif()

  # CycloneDDS C++ headers live under thirdparty/include/ddscxx (e.g.
  # ddscxx/dds/topic/TopicTraits.hpp), the C headers under thirdparty/include.
  add_library(motrix_unitree_sdk2 STATIC IMPORTED GLOBAL)
  set_target_properties(motrix_unitree_sdk2 PROPERTIES
    IMPORTED_LOCATION "${UNITREE_SDK2_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES
      "${UNITREE_SDK2_ROOT}/include;${UNITREE_SDK2_ROOT}/thirdparty/include;${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx")
  target_link_directories(motrix_unitree_sdk2 INTERFACE
    "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}")
  target_link_libraries(motrix_unitree_sdk2 INTERFACE ddscxx ddsc Threads::Threads dl)
  set(UNITREE_TARGET motrix_unitree_sdk2)
  set(UNITREE_RUNTIME_LIB_DIR "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}")
else()
  set(UNITREE_TARGET unitree_sdk2)
endif()

pybind11_add_module(_native src/backend.cpp src/bindings.cpp)
target_link_libraries(_native PRIVATE ${UNITREE_TARGET})
# DT_RPATH, not DT_RUNPATH: the robot shells source /opt/ros/foxy, whose
# LD_LIBRARY_PATH contains CycloneDDS 0.7.0. RUNPATH is searched after
# LD_LIBRARY_PATH, so the extension would bind to 0.7.0 and fail with
# "undefined symbol: ddsi_sertype_v0". RPATH is searched first.
target_link_options(_native PRIVATE "LINKER:--disable-new-dtags")

if(DEFINED UNITREE_RUNTIME_LIB_DIR)
  set_target_properties(_native PROPERTIES
    BUILD_RPATH "${UNITREE_RUNTIME_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN;${UNITREE_RUNTIME_LIB_DIR}")
else()
  # Installed SDK2 packages expose imported DDS targets. Ask CMake to carry
  # their link directories into the extension's runtime search path.
  set_target_properties(_native PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
install(TARGETS _native LIBRARY DESTINATION motrix_unitree_native)
