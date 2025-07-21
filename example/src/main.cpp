#include <iostream>

#include "lootinator/lootinator.h"
#include "lootinator/constraint/constraint.h"
#include "lootinator/"

int main() {
	std::vector<loot::Constraint> cons = loot::parse_constraints_from_json("../../lootinator/tests/constraints.json");
	loot::Constraint c = cons[0];
	// std::cout << c << "\n";
	std::cout << c.attributes << "\n";
	
	return 0;
}
