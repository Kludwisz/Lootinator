#include <iostream>
#include <algorithm>
#include "lootinator/assertions.h"
#include "lootinator/constraint/constraint.h"
#include "lootinator/utility/debug.h"

void test_json_parse() {
    try {
        std::vector<loot::Constraint> cons = loot::parse_constraints_from_json("../../lootinator/tests/constraints.json");

        std::vector<loot::ItemAttribute> attributes;
        attributes.push_back({1, {17, 37}});
        attributes.push_back({2, {21, 60}});
        
        loot::Constraint test_constraint = {10, {0, 15}, 1, attributes};
        loot::debug(std::cerr, attributes);

        ASSERT_EQ(cons[0], test_constraint);        
    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
	}
}

int LOOTINATOR_EXTERN tests_constraint_json_test(int argc, char** const argv) {
    test_json_parse();
    return 0;
}