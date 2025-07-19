#ifndef LOOTINATOR_CONSTRAINT_CONSTRAINT_H
#define LOOTINATOR_CONSTRAINT_CONSTRAINT_H

#include <cstdint>
#include <vector>
#include <ostream>
#include <algorithm>

namespace loot {
    constexpr int32_t UNUSED = -1;

    struct ItemAttribute {
        std::uint32_t type;
        std::uint32_t min_level;
        std::uint32_t max_level;

        bool operator==(const ItemAttribute& other) const {
            return type == other.type && min_level == other.min_level && max_level == other.max_level;
        }
    };

    // stores loot constraints on individual slots of items
    struct Constraint {
        std::uint32_t item;
        std::uint32_t min_count;
        std::uint32_t max_count;
        std::int32_t slot_id; // contraints are shared by cracking and finding kernels, finding won't use this

        std::vector<ItemAttribute> attributes;

        bool item_equal(const Constraint& other) const {
            if (item != other.item || attributes.size() != other.attributes.size()) 
                return false;

            // all item attributes must match
            return std::equal(attributes.begin(), attributes.begin() + attributes.size(), other.attributes.begin());
        }

        bool operator==(const Constraint& other) const {
            return item_equal(other) && other.min_count == min_count && other.max_count == max_count && other.slot_id == slot_id;
        }

        friend std::ostream& operator<<(std::ostream& os, const Constraint& constraint) {
            os << "Constraint{item=" << constraint.item << ", min=" << constraint.min_count 
                    << ", max=" << constraint.max_count << ", slot=" << constraint.slot_id << "}";
            return os;
        }
    };

    void merge_contraints(const std::vector<loot::Constraint>& src, std::vector<loot::Constraint>& dest);

    // static bool attributes_match(const std::vector<ItemAttribute>& first, const std::vector<ItemAttribute>& second) {
    //     for (const auto& e1 : first) {
    //         bool found = false;
    //         for (const auto& e2 : second) {
    //             if (e1.type == e2.type && e1.min_level == e2.min_level && e1.max_level == e2.max_level) {
    //                 found = true;
    //                 break;
    //             }
    //         }
    //         if (!found) {
    //             return false;
    //         }
    //     }

    //     return true;
    // }
}

#endif