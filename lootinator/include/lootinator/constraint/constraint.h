#ifndef LOOTINATOR_CONSTRAINT_CONSTRAINT_H
#define LOOTINATOR_CONSTRAINT_CONSTRAINT_H

#include "lootinator/utility/range.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <vector>
#include <ostream>
#include <algorithm>

namespace loot {
    constexpr int32_t SLOT_NONE = -1;

    struct ItemAttribute {
        std::uint32_t type;
        RangeInclusive<std::uint32_t> level_range;

        bool operator==(const ItemAttribute& other) const {
            return type == other.type && level_range == other.level_range;
        }

        static ItemAttribute from_json(nlohmann::json json) {
            uint32_t type = json["type"];
            loot::RangeInclusive<std::uint32_t> level_range = RangeInclusive<std::uint32_t>::from_json(json["level_range"]);
            return {type, level_range};
        }
    };

    inline bool attributes_match(const std::vector<ItemAttribute>& first, const std::vector<ItemAttribute>& second) {
        if (first.size() != second.size())
            return false;

        for (const auto& e1 : first) {
            bool found = false;
            for (const auto& e2 : second) {
                if (e1.type == e2.type && e1.level_range == e2.level_range) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

    // stores loot constraints on individual slots of items
    struct Constraint {
        std::uint32_t item;
        RangeInclusive<std::uint32_t> count_range;
        std::int32_t slot_id; // contraints are shared by cracking and finding kernels, finding won't use this

        std::vector<ItemAttribute> attributes;

        bool item_equal(const Constraint& other) const {
            if (item != other.item || attributes.size() != other.attributes.size()) 
                return false;

            // all item attributes must match
            return loot::attributes_match(attributes, other.attributes);
        }

        bool operator==(const Constraint& other) const {
            return other.count_range == count_range && other.slot_id == slot_id;
            //return item_equal(other) && other.count_range == count_range && other.slot_id == slot_id;
        }

        friend std::ostream& operator<<(std::ostream& os, const Constraint& constraint) {
            os << "Constraint{item=" << constraint.item << ", min=" << constraint.count_range.min 
                    << ", max=" << constraint.count_range.max << ", slot=" << constraint.slot_id << "}";
            return os;
        }
    };

    void merge_contraints(const std::vector<loot::Constraint>& src, std::vector<loot::Constraint>& dest);
    std::vector<loot::Constraint> parse_constraints_from_json(const char *filepath);
}

#endif
