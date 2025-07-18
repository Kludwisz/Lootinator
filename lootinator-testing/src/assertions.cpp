#include "lootinator-testing/assertions.h"

#include <exception>
#include <cstdlib>
#include <string>
#include <iostream>


namespace loottest {
    void assert_fail(const char* filename, const int line, std::string message) {
        std::cerr << filename << ':' << line << ' ' << message;
        //std::printf("%s:%d Test assertion failed: %s\n", filename, line, message);
        std::terminate();
    }


}
