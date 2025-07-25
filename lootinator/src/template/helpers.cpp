#include "lootinator/template/helpers.h"


namespace loot {
    void generate_set_seed(std::ostream& out, const std::string& prng_var_pointer, const std::string& seed_var) {
        out << '*' << prng_var_pointer << " = (0x5deece66dULL ^ " << seed_var << ") & ((1ULL<<48)-1);\n";
    }

    void generate_skip_n(std::ostream& out, const std::string& prng_var_pointer, const int64_t n) {
        uint64_t m = 1;
        uint64_t a = 0;
        uint64_t im = 0x5deece66dULL;
        uint64_t ia = 0xb;
        uint64_t k;

        for (k = static_cast<uint64_t>(n); k; k >>= 1)
        {
            if (k & 1)
            {
                m *= im;
                a = im * a + ia;
            }
            ia = (im + 1) * ia;
            im *= im;
        }

        m &= (1ULL << 48) - 1;
        a &= (1ULL << 48) - 1;

        out << '*' << prng_var_pointer << " = ((*" << prng_var_pointer << ") * " << m << "ULL + " << a << "ULL) & ((1ULL<<48)-1);\n"; 
    }

    void generate_set_count(std::ostream& out, const std::string& prng_var_pointer, const int min_inclusive, const int max_inclusive) {
        if (min_inclusive == max_inclusive) {
            out << min_inclusive << ";\n";
        }
        else {
            out << min_inclusive << " + " << "nextInt(" << prng_var_pointer << ", " << (max_inclusive - min_inclusive + 1) << ");\n";
        }
    }

    void generate_rarity_filter(std::ostream& out, const std::string& prng_var_pointer, const std::string& fail_operation, const float rarity) {
        // TODO
    }
}