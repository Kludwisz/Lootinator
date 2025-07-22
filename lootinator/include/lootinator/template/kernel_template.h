#ifndef LOOTINATOR_TEMPLATE_KERNEL_TEMPLATE_H
#define LOOTINATOR_TEMPLATE_KERNEL_TEMPLATE_H

#include "lootinator/template/template.h"


namespace loot {
    class KernelTemplate : Template {
    protected:
//        static const size_t DEFAULT_RESULT_ARRAY_CAPACITY = 1024 * 1024;

        std::string kernel_name;

        // these generators will be shared by all KernelTemplate objects
        void generate_loot_processors(std::ostream& out) const;
//        void generate_preamble(std::ostream& out) const;
//        void generate_loot_lookup_table(std::ostream& out) const;
        
        // these generators can, but may not be modified by each individual KernelTemplate subclass
        virtual void generate_kernel_header(std::ostream& out) const;
//        virtual void generate_kernel_launch_information(std::ostream &out) const;
//        virtual void generate_host_controller(std::ostream &out) const;

        // this generator defines the kernel structure and must be overriden by KernelTemplate subclasses.
        virtual void generate_kernel_body(std::ostream& out) const = 0;

    public:
        KernelTemplate(const TemplateParameters& params, const std::string& kernel_name);
        virtual void generate(std::ostream& out) const override;
    };
}

#endif