#include <iostream>
#include <algorithm>
#include "lootinator/assertions.h"
#include "lootinator/constraint/constraint.h"

void print_constraint_vec(std::vector<loot::Constraint> cvec) {
    for (auto& c : cvec)
        std::cerr << c << "\n";
    std::cerr << "-----\n";
}

void test_single_merge() {
    std::vector<loot::Constraint> constraints1;
    constraints1.push_back({2u, {1u, 3u}, 10});
    std::vector<loot::Constraint> constraints2;
    constraints2.push_back({2u, {2u, 5u}, 7});
    loot::merge_contraints(constraints1, constraints2);
    
    const loot::Constraint target = {2u, {3u, 8u}, loot::SLOT_NONE};
    const loot::Constraint actual = constraints2.at(0);
    ASSERT_EQ(target, actual);
}

// TODO make this use like a macro to avoid repetitions
void test_multi_merge() {
    std::vector<loot::Constraint> constraints1;
    std::vector<loot::Constraint> constraints2;

    // example constraint lists:
    std::vector<loot::ItemAttribute> c1_attrs;
    c1_attrs.push_back({0u, {2u,2u}});
    c1_attrs.push_back({2u, {1u,1u}});
    loot::Constraint c1 = {1u, {2u,3u}, 3, c1_attrs};
    std::vector<loot::ItemAttribute> c2_attrs;
    c2_attrs.push_back({0u, {2u,2u}});
    c2_attrs.push_back({2u, {1u,2u}});
    loot::Constraint c2 = {1u, {1u,4u}, 7, c2_attrs};

    std::vector<loot::ItemAttribute> c3_attrs;
    c3_attrs.push_back({2u, {1u,2u}});
    c3_attrs.push_back({0u, {2u,2u}});
    loot::Constraint c3 = {1u, {1u,1u}, 2, c3_attrs};
    std::vector<loot::ItemAttribute> c4_attrs;
    c4_attrs.push_back({2u, {3u,3u}});
    c4_attrs.push_back({3u, {1u,2u}});
    loot::Constraint c4 = {1u, {1u,1u}, 2, c4_attrs};

    constraints1.push_back(c1);
    constraints1.push_back(c2);
    constraints2.push_back(c3);
    constraints2.push_back(c4);
    print_constraint_vec(constraints1);
    print_constraint_vec(constraints2);

    loot::merge_contraints(constraints1, constraints2);
    print_constraint_vec(constraints2);

    std::vector<loot::Constraint> expectedConstraints;
    c1.slot_id = c3.slot_id = c4.slot_id = loot::SLOT_NONE;
    c3.count_range = {2u,5u};
    expectedConstraints.push_back(c1);
    expectedConstraints.push_back(c3);
    expectedConstraints.push_back(c4);
    print_constraint_vec(expectedConstraints);

    int equal_els = 0;
    for (const auto& con1 : constraints2) {
        for (const auto& con2 : expectedConstraints) {
            if (con1 == con2)
                equal_els++;
        }
    }
    ASSERT_EQ(equal_els, 3);
}

int LOOTINATOR_EXTERN tests_constraint_test(int argc, char** const argv) {
    test_single_merge();
    test_multi_merge();

    return 0;
}
