cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(ibow-lcd LANGUAGES CXX)

# Setting build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Setting the flags for profiling information or not
if(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Setting Release options")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Setting Debug options")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O1 -pg -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O1 -pg -march=native")
endif()

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.
                         Please use a different C++ compiler.")
endif()

# Printing the compiling flags
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

# Other packages
# find_package(OpenCV REQUIRED) # OpenCV
find_package(OpenCV 3.2)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenMP REQUIRED) # OpenMP
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


include_directories(include
                    external
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/thirdparty/obindex2/lib/include
                    ${PROJECT_SOURCE_DIR}/thirdparty/LOGOS/src
                    ${PROJECT_SOURCE_DIR}/thirdparty/GMS-Feature-Matcher/include
                    ${PROJECT_SOURCE_DIR}/thirdparty/lpm_cpp/src
                    )

add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdparty/obindex2/lib)

list(APPEND OBINDEX_LIBS  ${PROJECT_SOURCE_DIR}/thirdparty/obindex2/lib/build/libobindex2.a)

list(APPEND MATCHING_LIBS  
${PROJECT_SOURCE_DIR}/thirdparty/LOGOS/build/libLOGOS.a 
${PROJECT_SOURCE_DIR}/thirdparty/GMS-Feature-Matcher/build/libgms_match.a 
${PROJECT_SOURCE_DIR}/thirdparty/lpm_cpp/build/libLPM.a 
)
# Copy test images to the binary directory
file(GLOB JSON "evaluation/*.json")
file(COPY ${JSON}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

### Targets ###
# Library
add_library(lcdetector
            include/ibow-lcd/geometricconstraints.h
            src/geometricconstraints.cc
            include/ibow-lcd/island.h
            src/lcdetector.cc)
target_link_libraries(lcdetector
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${OBINDEX_LIBS}
                      )

# Main / Demo
add_executable(demo
               src/main.cc)
target_link_libraries(demo
                      lcdetector
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${OBINDEX_LIBS}
                      ${MATCHING_LIBS}
                      )

# Evaluation
add_executable(evaluator
               evaluation/lcevaluator.cc
               evaluation/main.cc)
target_link_libraries(evaluator
                      lcdetector
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${OBINDEX_LIBS}
                      ${MATCHING_LIBS}
                      )
