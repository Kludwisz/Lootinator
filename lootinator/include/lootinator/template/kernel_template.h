#ifndef LOOTINATOR_TEMPLATE_KERNEL_TEMPLATE_H
#define LOOTINATOR_TEMPLATE_KERNEL_TEMPLATE_H

#include "lootinator/template/template.h"


namespace loot {
    class KernelTemplate : Template {
    protected:
        static const size_t DEFAULT_RESULT_ARRAY_CAPACITY = 1024 * 1024;

        // these generators will be shared by all KernelTemplate objects
        void generatePreamble(std::ostream& out) const;
        void generateLootLookupTable(std::ostream& out) const;
        void generateLootProcessors(std::ostream& out) const;
        // these generators can be modified by each individual KernelTemplate subclass
        virtual void generateKernelHeader(std::ostream& out) const;
        virtual void generateKernelBody(std::ostream& out) const;
        virtual void generateHostController(std::ostream& out) const;

    public:
        KernelTemplate(const TemplateParameters& params);
        virtual void generate(std::ostream& out) const override;
    };
}

#endif