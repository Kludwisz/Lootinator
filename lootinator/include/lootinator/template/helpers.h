#ifndef LOOTINATOR_TEMPLATE_HELPERS_H
#define LOOTINATOR_TEMPLATE_HELPERS_H

#include <string>
#include <cstdint>
#include <ostream>

namespace loot {
    void generate_set_seed(std::ostream& out, const std::string& prng_var_pointer, const std::string& seed_var);
    void generate_skip_n(std::ostream& out, const std::string& prng_var_pointer, const int64_t n);
    void generate_set_count(std::ostream& out, const std::string& prng_var_pointer,const int min_inclusive, const int max_inclusive);
    void generate_rarity_filter(std::ostream& out, const std::string& prng_var_pointer, const std::string& fail_operation, const float rarity);
}

#endif