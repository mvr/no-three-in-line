enable_testing()

file(GLOB_RECURSE tests "*.cu")
# file(GLOB_RECURSE source "../src/*.cpp")

add_executable(three_tests ${tests} ${source})

include_directories("..")

target_link_libraries(three_tests gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(three_tests)
