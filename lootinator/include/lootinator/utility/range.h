#ifndef LOOTINATOR_UTILITY_RANGE_H
#define LOOTINATOR_UTILITY_RANGE_H

#include <nlohmann/json.hpp>

namespace loot {
    template <class T>
    struct RangeInclusive {
        T min;
        T max;

        RangeInclusive(T min, T max) : min(min), max(max) {}

        bool operator==(RangeInclusive other) const {
            return min == other.min && max == other.max;
        }

        bool operator!=(RangeInclusive other) const {
            return !(*this == other);
        }

        RangeInclusive merge(RangeInclusive other) const {
            return { min + other.min, max + other.max };
        }

        bool contains(T value) const {
            return value >= min && value <= max;
        }

        static RangeInclusive from_json(nlohmann::json json) {
            T min = json["min"];
            T max = json["max"];
            return { min, max };
        }
    };
}

#endif
