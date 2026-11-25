cmake_minimum_required(VERSION 3.16)
project(catchbox_franka LANGUAGES CXX)

option(CATCHBOX_BUILD_FRANKA "Build the libfranka hardware runner" ON)
include(CTest)
if(CATCHBOX_BUILD_FRANKA)
  find_package(Franka REQUIRED)
endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(FetchContent)
set(BUILD_EXAMPLES OFF)
set(BUILD_TESTS OFF)
set(BUILD_CLOUD_CLIENT OFF)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(ruckig
  URL https://codeload.github.com/pantor/ruckig/tar.gz/refs/tags/v0.19.4
  URL_HASH SHA256=4f57caa350770ce162aed92285fc07ac23e83b89bf802f5f500fb4bc8b2a1905)
FetchContent_MakeAvailable(ruckig)

if(CATCHBOX_BUILD_FRANKA)
  add_executable(franka_move franka_move.cpp)
  target_link_libraries(franka_move PRIVATE Franka::Franka Eigen3::Eigen ruckig::ruckig)
  if(BUILD_TESTING)
    add_test(NAME franka_move_help COMMAND franka_move --help)
    add_test(NAME franka_move_requires_center COMMAND franka_move --model fr3 --offset 0.01 0 0 --execute)
    set_tests_properties(franka_move_requires_center PROPERTIES
      PASS_REGULAR_EXPRESSION "--execute requires --center")
  endif()
endif()

if(BUILD_TESTING)
  add_executable(motion_plan_test motion_plan_test.cpp)
  target_link_libraries(motion_plan_test PRIVATE Eigen3::Eigen ruckig::ruckig)
  add_test(NAME motion_plan_test COMMAND motion_plan_test)
endif()
