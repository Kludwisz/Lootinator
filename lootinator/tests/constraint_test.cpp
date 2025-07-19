#include <iostream>
#include <algorithm>
#include "lootinator/assertions.h"
#include "lootinator/constraint/constraint.h"

void test_single_merge() {
    std::vector<loot::Constraint> constraints1;
    constraints1.push_back(loot::Constraint{2, 1, 3, 10});
    std::vector<loot::Constraint> constraints2;
    constraints2.push_back(loot::Constraint{2, 2, 5, 7});
    loot::merge_contraints(constraints1, constraints2);
    
    const loot::Constraint target = loot::Constraint{2, 3, 8, loot::UNUSED};
    const loot::Constraint actual = constraints2.at(0);
    ASSERT_EQ(target, actual);
}

void test_multi_merge() {
    std::vector<loot::Constraint> constraints1;
    std::vector<loot::Constraint> constraints2;
}

LOOTINATOR_EXTERN tests_constraint_test(int argc, char** const argv) {
    test_single_merge();

    return 0;
}