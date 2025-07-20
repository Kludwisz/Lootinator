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
        // here is the right spot for helper functions shared by all subclasses of Template:
        // - add array of constants to memory: either CPU, global, global-shared, managed, or constant 
        // - save loot seed result
        // ...

    public:
        Template(const TemplateParameters& params); // user input
        virtual std::string generate() const; // user output
    };
}

#endif