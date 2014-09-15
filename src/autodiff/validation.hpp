#ifndef nomad__src__autodiff__validation_hpp
#define nomad__src__autodiff__validation_hpp

#include <src/autodiff/exceptions.hpp>
#include <src/autodiff/autodiff_stack.hpp>

namespace nomad {

  template <std::string f_name>
  inline void validate_input(double val) {
    if (unlikely(std::isnan(input_val))) throw nomad_input_error(f_name);
    else return;
  }
  
  template <std::string f_name, double Lower>
  inline void validate_lower_bound(double val) {
    if (unlikely(input_val < Lower)) throw nomad_domain_error(f_name);
    else return;
  }
  
  template <std::string f_name, double Upper>
  inline void validate_upper_bound(double val) {
    if (unlikely(input_val > Upper)) throw nomad_domain_error(f_name);
    else return;
  }
  
  template <std::string f_name, double Lower, double Upper>
  inline void validate_lower_upper_bound(double val) {
    if (unlikely(input_val < Lower || input_val > Upper))
      throw nomad_domain_error(f_name);
    else return;
  }
  
}

#endif
