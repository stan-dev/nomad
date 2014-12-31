#ifndef nomad__src__validation__orderings_hpp
#define nomad__src__validation__orderings_hpp

#include <src/validation/exceptions.hpp>
#include <src/meta/unlikely.hpp>

namespace nomad {

  inline void validate_positive_ordering(double val1, double val2, std::string f_name) {
    if (unlikely(val1 < val2)) throw nomad_domain_error("positive-ordering", f_name);
    else return;
  }
    
}

#endif
