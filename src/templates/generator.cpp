#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <cinttypes>

/*
what we'll need:
- some form of text replacement patterns
- conditional replacement (handled generator-side)
- plain cuda code
*/


class TemplateBasedGenerator {
private:
    std::string plaintext;

public:
    TemplateBasedGenerator(const char* template_filename) {
        std::ifstream infile(template_filename);
        std::string file_contents { std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>() };
        this->plaintext = file_contents; // should call copy constr.
    }

    void process_symbol(std::string&& symbol, std::string&& value) {
        /*
        Two types of symbols?
        simple and conditional
        simple: inline declaration inside cuda code: @symbol@, marks that spot as replaceable by any given value
        conditional: @if:symbol@$code here?$
        */

        size_t pos = this->plaintext.find(symbol);
    }
};

int main() {
    TemplateBasedGenerator gentest("kernel.template");
    //gentest.print_plaintext();
    return 0;
}