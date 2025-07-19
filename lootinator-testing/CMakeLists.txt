include(CTest)
enable_testing()

add_library(lootinator-testing
    include/lootinator-testing/assertions.h
    src/assertions.cpp
)

target_include_directories(lootinator-testing PUBLIC include)
target_link_libraries(lootinator-testing PRIVATE lootinator)

set(TEST_FILES
    tests/constraint_test.cpp
)

# Create test driver and test source list
create_test_sourcelist(Tests CommonCxxTests.cxx ${TEST_FILES})

# Build test executable with the generated test driver
add_executable(test_driver ${Tests})
target_link_libraries(test_driver PRIVATE lootinator-testing)

foreach(test_file ${Tests})
    get_filename_component(test_name ${test_file} NAME_WE)
    # how do i get the full path of the tests' sources
    add_test(NAME ${test_name} COMMAND test_driver tests/${test_name})
endforeach()