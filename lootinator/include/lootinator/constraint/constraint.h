#ifndef LOOTINATOR_CONSTRAINT_CONSTRAINT_H
#define LOOTINATOR_CONSTRAINT_CONSTRAINT_H

#include "lootinator/utility/range.h"
#include "lootinator/utility/debug.h"

#include <nlohmann/json.hpp>

#include <cstdint>

namespace loot {
    constexpr int32_t SLOT_NONE = -1;

    // represents an additional attribute of an item, such as an enchantment
    // or type of music disc
    struct ItemAttribute {
        std::uint32_t type;
        RangeInclusive<std::uint32_t> level_range;

        static ItemAttribute from_json(nlohmann::json json) {
            uint32_t type = json["type"];
            loot::RangeInclusive<std::uint32_t> level_range = RangeInclusive<std::uint32_t>::from_json(json["level_range"]);
            return {type, level_range};
        }
        bool operator==(const ItemAttribute& other) const;
        bool operator!=(const ItemAttribute& other) const;
        friend std::ostream& operator<<(std::ostream& os, const ItemAttribute& attribute);
    };

    bool attributes_match(const std::vector<ItemAttribute>& first, const std::vector<ItemAttribute>& second);

    // stores loot constraints on individual slots of items
    struct Constraint {
        std::uint32_t item;
        RangeInclusive<std::uint32_t> count_range;
        std::int32_t slot_id; // contraints are shared by cracking and finding kernels, finding won't use this

        std::vector<ItemAttribute> attributes;

        bool item_equal(const Constraint& other) const;
        bool operator==(const Constraint& other) const;
        bool operator!=(const Constraint& other) const;
        friend std::ostream& operator<<(std::ostream& os, const Constraint& constraint);
    };

    void merge_contraints(const std::vector<loot::Constraint>& src, std::vector<loot::Constraint>& dest);
    std::vector<loot::Constraint> parse_constraints_from_json(const char *filepath);
}

#endif
