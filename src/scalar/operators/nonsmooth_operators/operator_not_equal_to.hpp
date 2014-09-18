#ifndef nomad__src__scalar__operators__nonsmooth_operators__operator_not_equal_to_hpp
#define nomad__src__scalar__operators__nonsmooth_operators__operator_not_equal_to_hpp

#include <src/var/var.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, bool >::type
    operator!=(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
               const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    if (ValidateIO) {
      validate_input(v1.first_val(), "operator!=");
      validate_input(v2.first_val(), "operator!=");
    }
    return v1.first_val() != v2.first_val();
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, bool >::type
    operator!=(double x,
               const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    if (ValidateIO) {
      validate_input(x, "operator!=");
      validate_input(v2.first_val(), "operator!=");
    }
    return x != v2.first_val();
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, bool >::type
    operator!=(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
               double y) {
    if (ValidateIO) {
      validate_input(v1.first_val(), "operator!=");
      validate_input(y, "operator!=");
    }
    return v1.first_val() != y;
  }

}
#endif
