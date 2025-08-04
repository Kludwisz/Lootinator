#ifndef LOOTINATOR_LOOT_TABLE_H
#define LOOTINATOR_LOOT_TABLE_H

#include "nlohmann/json.hpp"
#include <vector>
#include <string>

namespace loot {
	struct LootTable {
        nlohmann::json data;
        std::vector<std::string> item_names;
        std::vector<std::vector<int>> precomputed_loot;
        std::vector<int> total_weights;

        LootTable(const char* loot_table_json_filepath);

    private:
        void add_item_name(const std::string& item_name);
        int find_item_name(const std::string& item_name) const;
        void map_item_names();
        void precompute_entry_indices();
    };
}

#endif