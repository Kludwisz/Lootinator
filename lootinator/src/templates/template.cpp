#include "lootinator/template/template.h"

#include <sstream>


namespace loot {
    Template::Template(const TemplateParameters &params) {
        this->params = params;
    }

    std::string Template::generate() const {
        std::ostringstream oss;
        generate(oss);
        return oss.str();
    }
}

