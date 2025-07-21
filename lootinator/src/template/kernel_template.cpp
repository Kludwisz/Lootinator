#include "lootinator/template/kernel_template.h"

namespace loot {
    KernelTemplate::KernelTemplate(const TemplateParameters& params) : Template(params) {}

    void KernelTemplate::generate(std::ostream& out) const {
        generatePreamble(out);
        generateLootLookupTable(out);
        generateLootProcessors(out);
        generateKernelHeader(out);
        generateKernelBody(out);
        generateHostController(out);
    }

    void KernelTemplate::generatePreamble(std::ostream& out) const {
        // TODO
        // Generates a preamble containing code that will be shared by
        // all loot finding / loot cracking kernels:
        // 1. Includes for common CUDA / C++ headers. 
        // 2. GPU error handling macro 
        // 3. __managed__ memory storage for found loot seeds.
        // 4. __device__ PRNG functions for Java Random and Xoroshiro.
    }

    void KernelTemplate::generateLootLookupTable(std::ostream& out) const {
        // TODO
    }

    void KernelTemplate::generateLootProcessors(std::ostream& out) const {
        // TODO
    }

    void KernelTemplate::generateKernelHeader(std::ostream& out) const {
        // TODO
    }

    void KernelTemplate::generateKernelBody(std::ostream& out) const {
        // TODO
    }

    void KernelTemplate::generateHostController(std::ostream& out) const {
        // TODO
    }
}