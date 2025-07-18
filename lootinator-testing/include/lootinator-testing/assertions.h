#ifndef LOOTINATOR_TESTING_ASSERTIONS_H
#define LOOTINATOR_TESTING_ASSERTIONS_H

namespace loottest {
    template <class T, class U>
    void assert_compare_fail(const char* file, int line, const char* operator_symbol, const T& first, const U& second, const char* first_string, const char* second_string) {
        std::cerr << file << ":" << line << "Assertion \'" << operator_symbol << "\'failed!\n" 
                << first_string << " : " << first << "\n"
                << second_string << " : " << second << "\n";
    }

    #define ASSERT_COMPARE(first, operator_symbol, second) do {\
        const auto& first_var = first;\
        const auto& second_var = second;\
        if (!(first_var operator_symbol second_var)) {\
            loottest::assert_compare_fail(__FILE__, __LINE__, operator_symbol, first_var, second_var, #first, #second);\
        }\
    } while(false)

    // ----------------------------------------------

    #define ASSERT_EQ(first, second) ASSERT_COMPARE(first, ==, second)
    #define ASSERT_NE(first, second) ASSERT_COMPARE(first, !=, second)
    #define ASSERT_GT(first, second) ASSERT_COMPARE(first, >,  second)
    #define ASSERT_LT(first, second) ASSERT_COMPARE(first, <,  second)
    #define ASSERT_GE(first, second) ASSERT_COMPARE(first, >=, second)
    #define ASSERT_LE(first, second) ASSERT_COMPARE(first, <=, second)
}

#endif