#include "lootinator/constraint/constraint.h"

namespace loot {
    static void merge_into(std::vector<loot::Constraint>& dest, const loot::Constraint& constraint) {
        for (auto& stored_constraint : dest) {
            if (stored_constraint.item_equal(constraint)) {
                // have two item count ranges (min1, max1), (min2, max2)
                // the new min is min1+min2, new max is max1+max2
                stored_constraint.min_count += constraint.min_count;
                stored_constraint.max_count += constraint.max_count;
                stored_constraint.slot_id = loot::UNUSED;
                return;
            }
        }

        // did not find the item, create new constraint in the destination vector
        loot::Constraint to_add = constraint;
        to_add.slot_id = loot::UNUSED;
    }

    // accumulates the per-slot constraints into per-item-type ones (used by seedfinding kernels)
    // the acculumation takes into accout item enchantments
    void merge_contraints(const std::vector<loot::Constraint>& src, std::vector<loot::Constraint>& dest) {
        for (const auto& constraint : src) {
            merge_into(dest, constraint);
        }
    }
}

