#ifndef LOOTINATOR_TEMPLATES_TEMPLATE_H
#define LOOTINATOR_TEMPLATES_TEMPLATE_H

#include "lootinator/constraint/constraint.h"
#include <string>
#include <map>


namespace loot {
    struct TemplateParameters {
        std::vector<loot::Constraint> constraints;
        std::map<std::string, uint32_t> itemname_to_id;
        //LootTable loot_table;
        //MinecraftVersion version;
    };

    // Template objects are the main 
    class Template {
    protected:
        TemplateParameters params;
        // here is the right spot for helper functions shared by subclasses of Template:
        // - add array of constants to memory: either CPU, global, global-shared, managed, or constant 
        // - save loot seed result
        // ...

    public:
        Template(const TemplateParameters& params);

        // has to be defined by all subclasses of Template, generates and outputs
        // the entire piece of code to the out stream.
        virtual void generate(std::ostream& out) const = 0;

        // returns the result of generate(std::ostream&) as a string.
        std::string generate() const; 
    };
}

#endif