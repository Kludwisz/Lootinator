#include <iostream>

#include "lootinator/lootinator.h"
#include "lootinator/constraint/constraint.h"
#include "lootinator/utility/debug.h"

int main() {
	try {
		std::vector<loot::Constraint> cons = loot::parse_constraints_from_json("../../lootinator/tests/constraints.json");
		loot::Constraint c = cons[1];
		std::cout << c << "\n";
		loot::debug(std::cout, c.attributes);
		std::cout << "\n";
	}
	catch(...) {
		std::cout << "Error reading json file!\n";
	}
	
	return 0;
}
