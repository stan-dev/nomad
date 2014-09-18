#ifndef nomad__src__autodiff__validation_hpp
#define nomad__src__autodiff__validation_hpp

#include <src/autodiff/exceptions.hpp>
#include <src/autodiff/autodiff_stack.hpp>

namespace nomad {

  inline void validate_input(double val, std::string f_name) {
    if (unlikely(std::isnan(val))) throw nomad_input_error(f_name);
    else return;
  }
  
  inline void validate_lower_bound(double val, double lower, std::string f_name) {
    if (unlikely(val < lower)) throw nomad_domain_error(val, "lower bound", f_name);
    else return;
  }
  
  inline void validate_upper_bound(double val, double upper, std::string f_name) {
    if (unlikely(val > upper)) throw nomad_domain_error(val, "upper bound", f_name);
    else return;
  }
  
  inline void validate_positive_ordering(double val1, double val2, std::string f_name) {
    if (unlikely(val1 < val2)) throw nomad_domain_error("positive-ordering", f_name);
    else return;
  }
  
}

#endif
