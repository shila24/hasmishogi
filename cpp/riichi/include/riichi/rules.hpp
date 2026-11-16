#pragma once
#include "riichi/c_api.h"
namespace riichi {
// Retain integer flags so invalid ABI boolean values can be rejected.
struct Rules : rq_rules {
    Rules();
    explicit Rules(const rq_rules& value);
    void validate() const;
};
}
