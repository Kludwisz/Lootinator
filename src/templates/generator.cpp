#include <cstdio>
#include <string>
#include <cinttypes>

/*
what we'll need:
- some form of text replacement patterns
- conditional replacement (handled generator-side)
- plain cuda code
*/

class TemplateBasedGenerator {
private:
    char* plaintext;

public:
    TemplateBasedGenerator(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        fseek(fptr, SEEK_SET, SEEK_END);
        int file_size = ftell(fptr);
        printf("size: %d\n", file_size);
        fclose(fptr);

        this->plaintext = (char*)malloc((file_size + 1) * sizeof(char));
        fptr = fopen(filename, "r");
        fread(this->plaintext, sizeof(char), file_size, fptr);
        fclose(fptr);
    }

    void print_plaintext() {
        printf("%s\n", this->plaintext);
    }

    ~TemplateBasedGenerator() {
        free(plaintext);
    }
};

int main() {
    //FILE* fptr = fopen("test.json", "r");
    TemplateBasedGenerator gentest("test.json");
    gentest.print_plaintext();
    return 0;
}