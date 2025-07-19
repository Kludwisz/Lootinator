#include <iostream>
#include "lootinator-testing/assertions.h"

int __cdecl tests_constraint_test(int argc, char** const argv) {
    std::cout << "test1\n";
    int a = 1;
    int b = 2;
    ASSERT_EQ(a, b);
    return 0;
}