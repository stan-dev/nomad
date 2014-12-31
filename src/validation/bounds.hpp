#ifndef nomad__src__validation__bounds_hpp
#define nomad__src__validation__bounds_hpp

#include <src/validation/exceptions.hpp>
#include <src/meta/unlikely.hpp>

namespace nomad {

  inline void validate_lower_bound(double val, double lower, std::string f_name) {
    if (unlikely(val < lower)) throw nomad_domain_error(val, "lower bound", f_name);
    else return;
  }
  
  inline void validate_upper_bound(double val, double upper, std::string f_name) {
    if (unlikely(val > upper)) throw nomad_domain_error(val, "upper bound", f_name);
    else return;
  }
    
}

#endif
