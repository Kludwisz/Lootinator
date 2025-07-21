#ifndef LOOTINATOR_UTILITY_DEBUG_H
#define LOOTINATOR_UTILITY_DEBUG_H

#include <ostream>
#include <vector>

namespace loot {
    template <class T>
    std::ostream& debug(std::ostream& os, const T& value);

    template <class T>
    std::ostream& debug(std::ostream& os, const std::vector<T>& value);

    struct DebugArray {
        std::ostream* os;
        const char* delim = "";

        DebugArray(std::ostream& os);
        
        template <class T>
        DebugArray& add(const T& value) {
            debug(*os << delim, value);
            delim = ", ";
            return *this;
        }

        std::ostream& finish();
    };

    struct DebugStruct {
        std::ostream* os;
        const char* delim = "";

        DebugStruct(std::ostream& os, const char* name);

        template <class T>
        DebugStruct& add(const char* field, const T& value) {
            debug(*os << delim << field << "=", value);
            delim = ", ";
            return *this;
        }

        std::ostream& finish();
    };

    template <class T>
    std::ostream& debug(std::ostream& os, const T& value) {
        return os << value;
    }

    template <class T>
    std::ostream& debug(std::ostream& os, const std::vector<T>& vector) {
        DebugArray debug_array(os);

        for (const T& value : vector) {
            debug_array.add(value);
        }

        return debug_array.finish();
    }
}

#endif
