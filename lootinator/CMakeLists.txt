add_library(lootinator
    include/lootinator/assertions.h
    include/lootinator/lootinator.h
    include/lootinator/constraint/constraint.h
    include/lootinator/template/template.h
    include/lootinator/template/kernel_template.h
    include/lootinator/utility/debug.h
    include/lootinator/utility/range.h
    src/lootinator.cpp
    src/constraint/constraint.cpp
    src/template/template.cpp
    src/template/kernel_template.cpp
    src/utility/debug.cpp
)

target_include_directories(lootinator PUBLIC include)

set(TEST_FILES
    tests/constraint_test.cpp
)

# Create test driver and test source list
create_test_sourcelist(TEST_SOURCELIST CommonCxxTests.cxx ${TEST_FILES})

# Build test executable with the generated test driver
add_executable(test_driver ${TEST_SOURCELIST})
target_link_libraries(test_driver PRIVATE lootinator)

foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_test(NAME ${test_name} COMMAND test_driver tests/${test_name})
endforeach()
