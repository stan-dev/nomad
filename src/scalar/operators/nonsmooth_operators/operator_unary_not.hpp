#ifndef nomad__src__scalar__operators__nonsmooth_operators__operator_unary_not_hpp
#define nomad__src__scalar__operators__nonsmooth_operators__operator_unary_not_hpp

#include <src/var/var.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, bool >::type
    operator!(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    if (ValidateIO) validate_input(input.first_val(), "operator!");
    return !(input.first_val());
  }

}
#endif
