#include "loot_table.h"

#include <fstream>

namespace loot {
    LootTable::LootTable(const char* loot_table_json_filepath) {
        std::ifstream fin(loot_table_json_filepath);
        data = nlohmann::json::parse(fin);
        map_item_names();
        precompute_entry_indices();
    }

    void LootTable::add_item_name(const std::string& item_name) {
        if (find_item_name(item_name) == -1)
            item_names.push_back(item_name);
    }

    int LootTable::find_item_name(const std::string& item_name) const {
        int ix = 0;
        for (const std::string& stored_name : item_names) {
            if (stored_name == item_name) {
                return ix;
            }
            ix++;
        }
        return -1;
    }

    void LootTable::map_item_names() {
        const auto& pools = data["pools"];
        for (const auto& loot_pool : pools) {
            const auto& entries = data["entries"];
            for (const auto& entry : entries) {
                if (entry["type"] == "minecraft:item") {
                    add_item_name(entry["name"]);
                }
            }
        }
    }

    void LootTable::precompute_entry_indices() {
        const auto& pools = data["pools"];
        int pool_id = 0;
        for (const auto& loot_pool : pools) {
            int total_weight = 0;

            const auto& entries = data["entries"];
            for (const auto& entry : entries) {
                int entry_weight = 1;
                if (entry.contains("weight")) {
                    entry_weight = entry["weight"];
                }
                total_weight += entry_weight;

                // add the correct item id (or -1) to the precomputed loot table
                int iid = entry["type"] != "minecraft:item" ? -1 : find_item_name(entry["name"]);
                for (int i = 0; i < entry_weight; i++) {
                    precomputed_loot[pool_id].push_back(iid);
                }
            }

            pool_id++;
        }
    }
}
