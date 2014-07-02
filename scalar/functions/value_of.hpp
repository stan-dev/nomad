#ifndef nomad__scalar__functions__value_of_hpp
#define nomad__scalar__functions__value_of_hpp

#include <var/var.hpp>

namespace nomad {
  
  template <short autodiff_order>
  inline double value_of(const var<autodiff_order>& input) {
    return input.first_val();
  }

}

#endif
