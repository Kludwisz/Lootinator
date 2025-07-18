add_library(lootinator
	include/lootinator/lootinator.h
	include/lootinator/constraint/constraint.h
	src/lootinator.cpp
)

add_test(
	NAME constraint_merge_test
	COMMAND ...
)

target_include_directories(lootinator PUBLIC include)
