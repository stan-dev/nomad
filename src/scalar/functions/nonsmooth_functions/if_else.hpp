#ifndef nomad__src__scalar__functions__nonsmooth_functions__if_else_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__if_else_hpp

#include <type_traits>
#include <src/var/var.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
    if_else(bool c,
            const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v_true,
            const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v_false) {
    if (ValidateIO) {
      validate_input(v_true.first_val(), "if_else");
      validate_input(v_false.first_val(), "if_else");
    }
    return c ? v_true : v_false;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
  if_else(bool c,
          double x_true,
          const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v_false) {
    if (ValidateIO) {
      validate_input(x_true, "if_else");
      validate_input(v_false.first_val(), "if_else");
    }
    return c ? x_true : v_false;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
  if_else(bool c,
          const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v_true,
          double x_false) {
    if (ValidateIO) {
      validate_input(v_true.first_val(), "if_else");
      validate_input(x_false, "if_else");
    }
    return c ? v_true : x_false;
  }

}

#endif
