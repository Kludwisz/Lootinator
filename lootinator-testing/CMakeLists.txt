add_library(lootinator-testing
	include/lootinator-testing/assertions.h
	src/assertions.cpp
)

set(TEST_FILES
    tests/constraint/test1.cpp
    tests/constraint/test2.cpp
    tests/constraint/test3.cpp
)

create_test_sourcelist(TEST_SOURCELIST driver.cpp ${TEST_FILES})

add_executable(lootinator-test ${TEST_SOURCELIST})

foreach (test ${TEST_FILES})
    get_filename_component(test_name ${test} NAME_WE)
    add_test(NAME ${test_name} COMMAND lootinator-test ${test_name})
endforeach()

target_include_directories(lootinator-testing PUBLIC include)
target_link_libraries(example PRIVATE lootinator)
