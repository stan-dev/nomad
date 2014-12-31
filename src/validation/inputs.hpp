#ifndef nomad__src__validation__inputs_hpp
#define nomad__src__validation__inputs_hpp

#include <src/validation/exceptions.hpp>
#include <src/meta/unlikely.hpp>

namespace nomad {

  inline void validate_input(double val, std::string f_name) {
    if (unlikely(std::isnan(val))) throw nomad_input_error(f_name);
    else return;
  }
  
}

#endif
